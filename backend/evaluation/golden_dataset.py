import json
from pathlib import Path
from backend.llm_client import chat_completion_json
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.database.db import SessionLocal, MedicalDocument
from backend.medical_rag.rag_pipeline import medical_rag_query
from backend.logger import get_logger

logger = get_logger("evaluation.golden_dataset")

EVAL_DIR = Path("evaluation_results")
EVAL_DIR.mkdir(exist_ok=True)

# Static golden QA pairs — known correct answers
STATIC_GOLDEN_QA = [
    {
        "question": "What is the normal range for WBC count?",
        "ground_truth": "The normal range for WBC (White Blood Cell) count is 4.5 to 11.0 K/uL. Values above 11.0 K/uL indicate leukocytosis which may suggest infection, inflammation, or hematologic conditions.",
        "domain": "laboratory_medicine",
        "difficulty": "easy",
        "expected_terms": ["4.5", "11.0", "leukocytosis", "infection"],
    },
    {
        "question": "What is the first-line medication for Type 2 Diabetes?",
        "ground_truth": "Metformin is the first-line pharmacotherapy for Type 2 Diabetes. It works by inhibiting hepatic gluconeogenesis and improving insulin sensitivity. It is contraindicated in severe renal impairment (eGFR < 30).",
        "domain": "pharmacology",
        "difficulty": "easy",
        "expected_terms": ["metformin", "first-line", "glucose", "insulin"],
    },
    {
        "question": "What are the emergency signs of a cardiac event?",
        "ground_truth": "Emergency cardiac signs include chest pain or tightness, shortness of breath, sweating, nausea, left arm or jaw pain, and palpitations. These require immediate emergency services (911/112) and should not be managed with AI guidance.",
        "domain": "emergency_medicine",
        "difficulty": "medium",
        "expected_terms": ["chest pain", "emergency", "911", "immediate"],
    },
    {
        "question": "What monitoring is required for patients on Warfarin?",
        "ground_truth": "Patients on Warfarin require regular INR monitoring - weekly initially then monthly once stable. Target INR ranges vary by indication (typically 2.0-3.0 for AF, 2.5-3.5 for mechanical heart valves). Signs of bleeding should be monitored continuously.",
        "domain": "pharmacology",
        "difficulty": "medium",
        "expected_terms": ["inr", "monitoring", "bleeding", "warfarin"],
    },
    {
        "question": "What does an HbA1c of 7.5% indicate?",
        "ground_truth": "HbA1c of 7.5% indicates suboptimal diabetes control. The target for most diabetic patients is below 7.0%. At 7.5% the estimated average glucose is approximately 169 mg/dL. This level increases risk of microvascular complications and warrants medication review.",
        "domain": "endocrinology",
        "difficulty": "medium",
        "expected_terms": ["suboptimal", "7.0", "glucose", "complications"],
    },
    {
        "question": "What is the FAST acronym used for?",
        "ground_truth": "FAST is a stroke recognition acronym: F=Face drooping, A=Arm weakness, S=Speech difficulty, T=Time to call emergency services. Any FAST positive sign requires immediate 911/112 call as time is critical - every minute of stroke equals 1.9 million neurons lost.",
        "domain": "emergency_medicine",
        "difficulty": "easy",
        "expected_terms": ["face", "arm", "speech", "time", "emergency"],
    },
    {
        "question": "What are contraindications for Metformin?",
        "ground_truth": "Metformin contraindications include: eGFR < 30 mL/min/1.73m² (severe renal impairment), active liver disease, alcoholism, and iodinated contrast dye administration (hold 48 hours before and after). These contraindications relate to risk of lactic acidosis.",
        "domain": "pharmacology",
        "difficulty": "hard",
        "expected_terms": ["egfr", "renal", "contrast", "lactic acidosis"],
    },
    {
        "question": "What is the significance of a critical potassium level?",
        "ground_truth": "Critical potassium levels (below 2.5 or above 6.5 mEq/L) are medical emergencies. Hypokalemia causes muscle weakness, arrhythmias, and ECG changes. Hyperkalemia causes life-threatening cardiac arrhythmias and requires immediate treatment. Both require immediate physician notification.",
        "domain": "laboratory_medicine",
        "difficulty": "hard",
        "expected_terms": ["critical", "arrhythmia", "cardiac", "immediate"],
    },
]


def generate_document_golden_pairs(document_id: int) -> list[dict]:
    """
    Generates golden QA pairs specific to an uploaded document.
    Uses MiniMax to generate questions AND their correct answers
    from the actual document content.
    """
    logger.info(
        f"Generating document golden pairs | doc_id={document_id}"
    )

    db = SessionLocal()
    try:
        record = db.query(MedicalDocument).filter(
            MedicalDocument.id == document_id
        ).first()
        if not record:
            return []
        text = record.extracted_text[:2000]
        doc_type = record.document_type
    finally:
        db.close()

    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}

    diagnoses = [d.get("name") for d in entities.get("diagnoses", [])]
    labs = [l.get("test_name") for l in entities.get("lab_values", [])]
    meds = [m.get("name") for m in entities.get("medications", [])]

    prompt = f"""You are a medical education expert creating an evaluation dataset.

Based on this medical document, generate 5 question-answer pairs where
the answers are DIRECTLY AND COMPLETELY extractable from the document.

Document type: {doc_type}
Document content: {text}

Key findings:
- Diagnoses: {', '.join(diagnoses[:3]) or 'None'}
- Lab values: {', '.join(labs[:3]) or 'None'}
- Medications: {', '.join(meds[:3]) or 'None'}

Return ONLY valid JSON array:
[
    {{
        "question": "Specific question answerable from document",
        "ground_truth": "Complete, accurate answer from document content",
        "domain": "laboratory_medicine|pharmacology|clinical_note|diagnosis",
        "difficulty": "easy|medium|hard",
        "expected_terms": ["key", "terms", "that", "should", "appear"]
    }}
]

Requirements:
- Questions must be answerable ONLY from this document
- Ground truth must be factually accurate
- Include specific values, names, dosages from document
- Return valid JSON only"""

    try:
        raw = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical evaluation expert. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
        )

        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        pairs = json.loads(cleaned.strip())
        if isinstance(pairs, list):
            logger.info(
                f"Generated {len(pairs)} golden pairs | doc_id={document_id}"
            )
            return pairs
        return []

    except Exception as e:
        logger.error(f"Golden pair generation failed: {e}")
        return []


def build_golden_dataset(
    document_id: int = None,
    include_static: bool = True,
) -> dict:
    """
    Builds complete golden evaluation dataset.
    Combines static QA pairs + document-specific pairs.
    """
    logger.info("Building golden evaluation dataset")
    all_pairs = []

    if include_static:
        all_pairs.extend(STATIC_GOLDEN_QA)
        logger.info(f"Added {len(STATIC_GOLDEN_QA)} static golden pairs")

    if document_id:
        doc_pairs = generate_document_golden_pairs(document_id)
        all_pairs.extend(doc_pairs)
        logger.info(
            f"Added {len(doc_pairs)} document-specific pairs"
        )

    # Save dataset
    dataset_path = EVAL_DIR / f"golden_dataset_doc{document_id or 'static'}.json"
    with open(dataset_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    return {
        "total_pairs": len(all_pairs),
        "static_pairs": len(STATIC_GOLDEN_QA) if include_static else 0,
        "document_pairs": len(all_pairs) - (
            len(STATIC_GOLDEN_QA) if include_static else 0
        ),
        "domains": list(set(p.get("domain", "general") for p in all_pairs)),
        "difficulties": {
            "easy": sum(1 for p in all_pairs if p.get("difficulty") == "easy"),
            "medium": sum(1 for p in all_pairs if p.get("difficulty") == "medium"),
            "hard": sum(1 for p in all_pairs if p.get("difficulty") == "hard"),
        },
        "saved_to": str(dataset_path),
    }