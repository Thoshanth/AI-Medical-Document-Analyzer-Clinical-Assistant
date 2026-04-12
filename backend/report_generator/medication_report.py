from backend.llm_client import chat_completion
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.drug_interaction.interaction_pipeline import check_all_medications
from backend.knowledge_base.drug_db import get_drug_info
from backend.logger import get_logger
from datetime import datetime

logger = get_logger("report_generator.medication")


def generate_medication_report(document_id: int) -> dict:
    """
    Generates a pharmacist-style medication review report.

    Combines:
    - Stage 2: extracted medications with dosages
    - Stage 3: drug knowledge base (contraindications, monitoring)
    - Stage 5: drug interaction findings
    """
    logger.info(
        f"Generating medication report | doc_id={document_id}"
    )

    entities_data = get_clinical_entities(document_id)
    if not entities_data:
        raise ValueError(
            f"No entities for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )

    medications = entities_data.get(
        "entities", {}
    ).get("medications", [])

    if not medications:
        return {
            "report_type": "medication_review",
            "document_id": document_id,
            "generated_at": datetime.utcnow().isoformat(),
            "report": "No medications found in this document.",
            "interactions": [],
            "disclaimer": "",
        }

    # Get drug interactions from Stage 5
    interactions_data = {}
    try:
        interactions_data = check_all_medications(document_id)
    except Exception as e:
        logger.warning(f"Interaction check failed: {e}")

    # Get KB drug info for each medication
    drug_details = []
    for med in medications:
        name = med.get("name", "")
        info = get_drug_info(name)
        drug_details.append({
            "name": name,
            "dosage": med.get("dosage", "Not specified"),
            "frequency": med.get(
                "frequency_normalized",
                med.get("frequency", "Not specified")
            ),
            "drug_class": info.get("drug_class", "Unknown") if info else "Unknown",
            "high_risk": med.get("high_risk", False),
            "contraindications": info.get(
                "contraindications", []
            )[:3] if info else [],
            "monitoring": info.get("monitoring", [])[:3] if info else [],
        })

    # Build prompt context
    medications_detail = "\n".join([
        f"{i+1}. {d['name']} {d['dosage']} {d['frequency']}\n"
        f"   Class: {d['drug_class']}\n"
        f"   High-risk: {'YES ⚠️' if d['high_risk'] else 'No'}\n"
        f"   Contraindications: {', '.join(d['contraindications']) or 'None noted'}\n"
        f"   Monitoring: {', '.join(d['monitoring']) or 'Standard monitoring'}"
        for i, d in enumerate(drug_details)
    ])

    interactions = interactions_data.get("interactions", [])
    interaction_summary = "\n".join([
        f"- {i.get('drug_a')} + {i.get('drug_b')}: "
        f"[{i.get('severity','').upper()}] {i.get('effect','')[:100]}"
        for i in interactions[:5]
    ]) or "No significant interactions identified"

    system_prompt = """You are a senior clinical pharmacist conducting 
a medication review. Be thorough, systematic, and safety-focused.
Flag all concerns clearly."""

    user_prompt = f"""Generate a comprehensive medication review report.

CURRENT MEDICATIONS:
{medications_detail}

DRUG INTERACTIONS IDENTIFIED:
{interaction_summary}

Generate a pharmacist medication review with EXACTLY these sections:

**MEDICATION REVIEW SUMMARY:**
[Overview of medication regimen]

**INDIVIDUAL MEDICATION ASSESSMENT:**
[For each medication: appropriateness, dosing, monitoring needs]

**DRUG INTERACTION ANALYSIS:**
[Detailed review of all identified interactions with severity]

**HIGH-RISK MEDICATION ALERTS:**
[Special attention items for dangerous medications]

**MONITORING REQUIREMENTS:**
[Specific lab tests and clinical parameters to monitor]

**PHARMACIST RECOMMENDATIONS:**
[Actionable recommendations for prescriber]

**PATIENT COUNSELING POINTS:**
[Key information patient should know]

Be specific and clinically precise."""

    report_text = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.1,
    )

    logger.info(
        f"Medication report generated | chars={len(report_text)}"
    )

    return {
        "report_type": "medication_review",
        "document_id": document_id,
        "generated_at": datetime.utcnow().isoformat(),
        "report": report_text,
        "interactions_found": len(interactions),
        "major_interactions": sum(
            1 for i in interactions if i.get("severity") == "major"
        ),
        "high_risk_medications": [
            d["name"] for d in drug_details if d["high_risk"]
        ],
        "metadata": {
            "total_medications": len(medications),
            "interactions_checked": interactions_data.get(
                "pairs_checked", 0
            ),
        },
        "disclaimer": (
            "⚕️ This medication review is AI-generated. "
            "All recommendations must be verified by a licensed "
            "pharmacist or physician before implementation."
        ),
    }