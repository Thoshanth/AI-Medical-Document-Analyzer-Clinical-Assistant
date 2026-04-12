import json
import random
from pathlib import Path
from backend.llm_client import chat_completion_json
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.database.db import SessionLocal, MedicalDocument
from backend.knowledge_base.disease_db import DISEASE_DATABASE
from backend.knowledge_base.drug_db import LOCAL_DRUG_DB
from backend.logger import get_logger

logger = get_logger("fine_tuning.dataset")

FT_DATA_DIR = Path("fine_tuning_data")
FT_DATA_DIR.mkdir(exist_ok=True)

# Medical system prompt used for all training examples
MEDICAL_SYSTEM_PROMPT = (
    "You are an expert medical AI assistant trained on clinical "
    "literature. Provide accurate, evidence-based medical information. "
    "Always recommend consulting healthcare professionals for "
    "clinical decisions."
)

# Template QA pairs from our medical knowledge base
# These are high-quality curated examples
TEMPLATE_QA_PAIRS = [
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What is the first-line treatment for Type 2 Diabetes?",
        "output": (
            "Metformin is the first-line pharmacotherapy for Type 2 "
            "Diabetes Mellitus (T2DM). It belongs to the biguanide class "
            "and works by inhibiting hepatic gluconeogenesis and improving "
            "peripheral insulin sensitivity. Standard dosing starts at "
            "500mg twice daily with meals, titrated to 1000-2000mg/day. "
            "Lifestyle modification including diet control and regular "
            "exercise are equally important first-line interventions. "
            "Contraindications include eGFR < 30 mL/min/1.73m², "
            "hepatic impairment, and use of iodinated contrast dye. "
            "HbA1c should be monitored every 3 months to assess response."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What are the emergency signs in a diabetic patient?",
        "output": (
            "Emergency signs in diabetic patients include: "
            "1) Diabetic Ketoacidosis (DKA): blood glucose > 250 mg/dL, "
            "ketones in urine, fruity breath, vomiting, abdominal pain, "
            "Kussmaul breathing. Requires immediate hospitalization and "
            "IV insulin protocol. "
            "2) Hyperosmolar Hyperglycemic State (HHS): glucose > 600 mg/dL, "
            "severe dehydration, altered consciousness, no significant ketosis. "
            "3) Severe Hypoglycemia: glucose < 54 mg/dL with altered "
            "consciousness — give IV dextrose or glucagon. "
            "4) Signs of end-organ damage: chest pain (MI), severe headache "
            "(stroke), acute vision loss (retinal emergency). "
            "All these require immediate emergency medical attention."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "Explain the Warfarin-Aspirin drug interaction.",
        "output": (
            "The Warfarin-Aspirin interaction is a major drug interaction "
            "with potentially life-threatening consequences. "
            "Mechanism: Pharmacodynamic interaction — both drugs affect "
            "hemostasis through different pathways. Warfarin inhibits "
            "vitamin K-dependent clotting factors (II, VII, IX, X) while "
            "aspirin inhibits platelet aggregation via COX-1 inhibition "
            "and causes gastric mucosal damage. "
            "Effect: Combined use increases bleeding risk 3-5x compared "
            "to warfarin alone. Risk of serious GI hemorrhage, "
            "intracranial bleeding, and other major bleeds. "
            "Management: Avoid combination when possible. If clinically "
            "necessary (e.g., post-MI with AF), use lowest effective "
            "aspirin dose (75-100mg), monitor INR closely, use proton "
            "pump inhibitor for GI protection, and educate patient on "
            "bleeding warning signs."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "How do you interpret an elevated WBC count?",
        "output": (
            "White Blood Cell (WBC) count elevation (leukocytosis) "
            "requires clinical correlation for proper interpretation. "
            "Normal range: 4.5-11.0 K/uL. "
            "Mild elevation (11-20 K/uL): Common causes include bacterial "
            "infection, inflammation, stress response, steroid use, "
            "or vigorous exercise. "
            "Moderate elevation (20-50 K/uL): Suggests significant "
            "infection (sepsis), severe inflammatory conditions, "
            "leukemoid reaction, or hematologic malignancy. "
            "Severe elevation (> 50 K/uL): Consider leukemia, "
            "leukemoid reaction, or severe systemic infection. "
            "Differential count is critical: neutrophilia suggests "
            "bacterial infection; lymphocytosis suggests viral infection; "
            "eosinophilia suggests allergy or parasites; "
            "monocytosis suggests chronic infection. "
            "Always correlate with clinical presentation, fever, "
            "CRP, procalcitonin, and clinical symptoms."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What is the FAST acronym for stroke recognition?",
        "output": (
            "FAST is the mnemonic for rapid stroke recognition: "
            "F — Face: Ask patient to smile. Facial drooping or asymmetry "
            "suggests stroke. "
            "A — Arms: Ask patient to raise both arms. Arm weakness or "
            "drift on one side is a stroke sign. "
            "S — Speech: Ask patient to repeat a simple phrase. Slurred, "
            "strange, or absent speech indicates stroke. "
            "T — Time: TIME IS BRAIN. Every minute of stroke = 1.9 million "
            "neurons lost. Call emergency services immediately. "
            "Note time of symptom onset — crucial for thrombolysis decision "
            "(tPA window: 3-4.5 hours from onset). "
            "Additional signs: sudden severe headache, vision changes, "
            "dizziness, sudden confusion, loss of balance. "
            "Any FAST positive finding requires immediate 911/112 call "
            "and transport to stroke center."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What are the components of a SOAP note?",
        "output": (
            "A SOAP note is the standard clinical documentation format: "
            "S — SUBJECTIVE: Patient-reported information. "
            "Chief complaint (CC), history of present illness (HPI), "
            "review of systems (ROS), past medical history (PMH), "
            "medications, allergies, social history, family history. "
            "O — OBJECTIVE: Clinician-measured findings. "
            "Vital signs, physical examination findings, laboratory "
            "results, imaging results, ECG findings. "
            "A — ASSESSMENT: Clinical interpretation. "
            "Primary diagnosis with ICD-10 code, differential diagnoses, "
            "problem list, severity assessment. "
            "P — PLAN: Management decisions. "
            "Medications prescribed (with dose and duration), "
            "investigations ordered, referrals, procedures planned, "
            "patient education, follow-up schedule. "
            "SOAP notes provide structured, reproducible documentation "
            "that facilitates communication between healthcare providers "
            "and supports continuity of care."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What does an elevated HbA1c of 8.5% indicate?",
        "output": (
            "HbA1c of 8.5% indicates poorly controlled diabetes. "
            "Interpretation: HbA1c < 5.7% = Normal, 5.7-6.4% = Prediabetes, "
            "≥ 6.5% = Diabetes diagnosis, < 7.0% = Well-controlled diabetes target, "
            "7.0-8.0% = Suboptimal control, > 8.0% = Poor control. "
            "At 8.5%, the estimated average glucose (eAG) is approximately "
            "197 mg/dL. This level significantly increases risk of: "
            "Microvascular complications (retinopathy, nephropathy, neuropathy) "
            "and macrovascular complications (cardiovascular disease, stroke). "
            "Management: Review current medication regimen, assess adherence, "
            "consider intensification of therapy (add second agent or increase "
            "Metformin dose), dietary counseling, increase monitoring frequency. "
            "Target: < 7.0% for most patients (individualize for elderly/comorbid). "
            "Recheck HbA1c in 3 months after therapy change."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What is the mechanism of ACE inhibitors?",
        "output": (
            "ACE inhibitors (e.g., Lisinopril, Enalapril, Ramipril) work by: "
            "1) Blocking Angiotensin-Converting Enzyme (ACE) which normally "
            "converts Angiotensin I → Angiotensin II. "
            "2) Reduced Angiotensin II leads to: vasodilation (reduced "
            "peripheral resistance), decreased aldosterone secretion "
            "(reduced sodium and water retention), reduced preload and "
            "afterload on the heart. "
            "3) Increased bradykinin (ACE normally degrades bradykinin) "
            "contributes to vasodilation and causes the classic dry cough "
            "side effect (in 10-15% of patients). "
            "Clinical uses: Hypertension (first-line), Heart failure "
            "(reduces mortality), Diabetic nephropathy (renoprotective), "
            "Post-MI (reduces remodeling). "
            "Key monitoring: Serum potassium (hyperkalemia risk), "
            "creatinine (watch for acute kidney injury, especially first "
            "2 weeks), blood pressure. "
            "Contraindications: Pregnancy (teratogenic — Category D), "
            "bilateral renal artery stenosis, history of angioedema."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "How do you interpret a chest X-ray for pneumonia?",
        "output": (
            "Chest X-ray (CXR) findings in pneumonia: "
            "Classic consolidation: Homogeneous opacity in one or more "
            "lobes with air bronchograms (visible bronchi within opacity). "
            "Lobar pneumonia: Dense consolidation of entire lobe — "
            "typically bacterial (Streptococcus pneumoniae). "
            "Bronchopneumonia: Patchy bilateral perihilar infiltrates — "
            "typical in hospital-acquired and aspiration pneumonia. "
            "Interstitial pattern: Bilateral reticular/ground-glass "
            "opacities — suggests atypical pneumonia (Mycoplasma, "
            "Legionella, viral) or COVID-19. "
            "Associated findings to note: Pleural effusion (parapneumonic "
            "effusion or empyema), cavitation (Klebsiella, TB, abscess), "
            "air bronchograms, volume loss. "
            "CXR may be normal in early pneumonia or immunocompromised "
            "patients — CT chest is more sensitive. "
            "Follow-up CXR at 6 weeks recommended to ensure resolution "
            "and exclude underlying malignancy."
        ),
    },
    {
        "instruction": MEDICAL_SYSTEM_PROMPT,
        "input": "What is the difference between sensitivity and specificity?",
        "output": (
            "Sensitivity and specificity are fundamental test characteristics: "
            "SENSITIVITY (True Positive Rate): "
            "Ability to correctly identify people WITH the disease. "
            "Formula: TP / (TP + FN). "
            "High sensitivity = few false negatives = good RULING OUT test. "
            "Mnemonic: SnNOut — high Sensitivity, Negative result rules OUT. "
            "Use when: Missing a diagnosis is dangerous "
            "(e.g., HIV screening, PE rule-out). "
            "SPECIFICITY (True Negative Rate): "
            "Ability to correctly identify people WITHOUT the disease. "
            "Formula: TN / (TN + FP). "
            "High specificity = few false positives = good RULING IN test. "
            "Mnemonic: SpPIn — high Specificity, Positive result rules IN. "
            "Use when: False positives cause harm "
            "(e.g., confirmatory HIV test, cancer biopsy). "
            "Clinical application: Screening tests prioritize sensitivity; "
            "confirmatory tests prioritize specificity. "
            "PPV and NPV depend on disease prevalence in the population."
        ),
    },
]


def generate_qa_from_document(document_id: int) -> list[dict]:
    """
    Generates medical QA pairs from an uploaded document
    using MiniMax to create training examples.

    Each document generates 5-10 QA pairs covering:
    - Key diagnoses found
    - Lab value interpretations
    - Medication questions
    - Clinical reasoning questions
    """
    logger.info(f"Generating QA pairs | doc_id={document_id}")

    # Load document
    db = SessionLocal()
    try:
        record = db.query(MedicalDocument).filter(
            MedicalDocument.id == document_id
        ).first()
        if not record:
            return []
        text = record.extracted_text[:3000]
        doc_type = record.document_type
    finally:
        db.close()

    # Load clinical entities
    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}

    diagnoses = entities.get("diagnoses", [])
    medications = entities.get("medications", [])
    lab_values = entities.get("lab_values", [])

    prompt = f"""Generate 5 medical question-answer pairs from this clinical document.

Document type: {doc_type}
Document content: {text}

Key clinical findings:
- Diagnoses: {', '.join([d.get('name','') for d in diagnoses[:3]]) or 'None'}
- Medications: {', '.join([m.get('name','') for m in medications[:3]]) or 'None'}
- Abnormal labs: {', '.join([l.get('test_name','') for l in lab_values if l.get('interpretation') != 'normal'][:3]) or 'None'}

Return ONLY valid JSON array:
[
    {{
        "instruction": "You are an expert medical AI assistant. Provide accurate, evidence-based medical information.",
        "input": "Medical question about the document content",
        "output": "Comprehensive, clinically accurate answer (3-5 sentences minimum)"
    }}
]

Rules:
- Questions must be answerable from the document content
- Answers must be clinically accurate and detailed
- Cover different aspects: diagnosis, treatment, lab interpretation, medications
- Return valid JSON array only"""

    try:
        raw = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical education expert. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
        )

        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        qa_pairs = json.loads(cleaned.strip())

        if isinstance(qa_pairs, list):
            logger.info(
                f"Generated {len(qa_pairs)} QA pairs | doc_id={document_id}"
            )
            return qa_pairs
        return []

    except Exception as e:
        logger.error(f"QA generation failed: {e}")
        return []


def generate_kb_qa_pairs() -> list[dict]:
    """
    Generates QA pairs from our Stage 3 knowledge base
    (disease database and drug database).
    Creates comprehensive training examples covering all conditions.
    """
    logger.info("Generating knowledge base QA pairs")
    pairs = []

    # Disease-based QA pairs
    for disease_name, info in DISEASE_DATABASE.items():
        # Treatment question
        pairs.append({
            "instruction": MEDICAL_SYSTEM_PROMPT,
            "input": f"What is the standard treatment for {info.get('full_name', disease_name)}?",
            "output": (
                f"{info.get('full_name', disease_name)} treatment: "
                f"First-line: {', '.join(info.get('first_line_treatments', [])[:3])}. "
                f"Second-line options include: "
                f"{', '.join(info.get('second_line_treatments', [])[:2]) or 'specialist referral'}. "
                f"Monitoring required: {', '.join(info.get('monitoring', [])[:3])}. "
                f"Emergency signs: {', '.join(info.get('emergency_signs', [])[:2])}."
            ),
        })

        # Symptoms question
        pairs.append({
            "instruction": MEDICAL_SYSTEM_PROMPT,
            "input": f"What are the classic symptoms of {info.get('full_name', disease_name)}?",
            "output": (
                f"Classic symptoms of {info.get('full_name', disease_name)}: "
                f"{', '.join(info.get('symptoms', [])[:6])}. "
                f"Risk factors include: "
                f"{', '.join(info.get('risk_factors', [])[:3])}. "
                f"Potential complications: "
                f"{', '.join(info.get('complications', [])[:3])}."
            ),
        })

        # Complications question
        if info.get("complications"):
            pairs.append({
                "instruction": MEDICAL_SYSTEM_PROMPT,
                "input": (
                    f"What complications should be monitored in a patient "
                    f"with {info.get('full_name', disease_name)}?"
                ),
                "output": (
                    f"Key complications of "
                    f"{info.get('full_name', disease_name)}: "
                    f"{', '.join(info.get('complications', []))}. "
                    f"Monitoring protocol: "
                    f"{', '.join(info.get('monitoring', [])[:4])}. "
                    f"Emergency signs requiring immediate attention: "
                    f"{', '.join(info.get('emergency_signs', [])[:3])}."
                ),
            })

    # Drug-based QA pairs
    for drug_name, info in LOCAL_DRUG_DB.items():
        pairs.append({
            "instruction": MEDICAL_SYSTEM_PROMPT,
            "input": f"What are the key clinical considerations for {drug_name.title()}?",
            "output": (
                f"{drug_name.title()} ({info.get('drug_class', 'medication')}): "
                f"Indicated for: {', '.join(info.get('indications', [])[:2])}. "
                f"Common side effects: {', '.join(info.get('side_effects', [])[:3])}. "
                f"Contraindications: {', '.join(info.get('contraindications', [])[:2])}. "
                f"Monitoring: {', '.join(info.get('monitoring', [])[:2])}. "
                f"{'HIGH-RISK medication requiring extra vigilance.' if info.get('high_risk') else 'Standard monitoring applies.'}"
            ),
        })

    logger.info(f"Generated {len(pairs)} knowledge base QA pairs")
    return pairs


def build_training_dataset(
    document_ids: list[int] = None,
    include_templates: bool = True,
    include_kb: bool = True,
) -> dict:
    """
    Builds complete training dataset combining:
    1. Curated template QA pairs (high quality baseline)
    2. Knowledge base QA pairs (disease + drug knowledge)
    3. Document-specific QA pairs (from uploaded documents)

    Splits into train (80%) and eval (20%) sets.
    Saves to fine_tuning_data/ directory.
    """
    logger.info("Building medical fine-tuning dataset")
    all_pairs = []

    # Source 1: Template QA pairs
    if include_templates:
        all_pairs.extend(TEMPLATE_QA_PAIRS)
        logger.info(f"Added {len(TEMPLATE_QA_PAIRS)} template pairs")

    # Source 2: Knowledge base pairs
    if include_kb:
        kb_pairs = generate_kb_qa_pairs()
        all_pairs.extend(kb_pairs)
        logger.info(f"Added {len(kb_pairs)} KB pairs")

    # Source 3: Document-specific pairs
    if document_ids:
        for doc_id in document_ids:
            doc_pairs = generate_qa_from_document(doc_id)
            all_pairs.extend(doc_pairs)
            logger.info(
                f"Added {len(doc_pairs)} pairs from doc {doc_id}"
            )

    # Shuffle for better training
    random.shuffle(all_pairs)

    # Split 80/20 train/eval
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]

    # Save datasets
    train_path = FT_DATA_DIR / "train_dataset.json"
    eval_path = FT_DATA_DIR / "eval_dataset.json"

    with open(train_path, "w") as f:
        json.dump(train_pairs, f, indent=2)

    with open(eval_path, "w") as f:
        json.dump(eval_pairs, f, indent=2)

    logger.info(
        f"Dataset built | "
        f"total={len(all_pairs)} | "
        f"train={len(train_pairs)} | "
        f"eval={len(eval_pairs)}"
    )

    return {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "sources": {
            "templates": len(TEMPLATE_QA_PAIRS) if include_templates else 0,
            "knowledge_base": len(kb_pairs) if include_kb else 0,
            "documents": len(all_pairs) - len(TEMPLATE_QA_PAIRS) - (len(kb_pairs) if include_kb else 0),
        },
    }