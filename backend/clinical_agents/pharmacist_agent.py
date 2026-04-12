from backend.llm_client import chat_completion
from backend.drug_interaction.interaction_pipeline import check_all_medications
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.knowledge_base.drug_db import get_drug_info
from backend.logger import get_logger

logger = get_logger("clinical_agents.pharmacist")


def pharmacist_agent(state: dict) -> dict:
    """
    Agent 3 — Pharmacist Agent

    Medication safety specialist. Reviews all medications,
    checks interactions (Stage 5), verifies dosing and
    contraindications.

    Reads:  document_id, diagnosis_report, primary_diagnosis
    Writes: pharmacist_report, drug_alerts, interaction_summary
    """
    document_id = state["document_id"]
    question = state["patient_question"]
    primary_diagnosis = state.get("primary_diagnosis", "")
    iteration = state.get("iterations", 1)

    logger.info(
        f"Pharmacist Agent | doc_id={document_id} | iter={iteration}"
    )

    # Load medications
    entities_data = get_clinical_entities(document_id)
    medications = entities_data.get(
        "entities", {}
    ).get("medications", []) if entities_data else []

    # Get drug interactions from Stage 5
    interactions_data = {}
    drug_alerts = []

    try:
        interactions_data = check_all_medications(document_id)
        interactions = interactions_data.get("interactions", [])

        for interaction in interactions:
            if interaction.get("severity") in ["major", "moderate"]:
                drug_alerts.append({
                    "type": "drug_interaction",
                    "severity": interaction.get("severity"),
                    "drugs": f"{interaction.get('drug_a')} + {interaction.get('drug_b')}",
                    "effect": interaction.get("effect", "")[:100],
                    "management": interaction.get("management", "")[:100],
                })
    except Exception as e:
        logger.warning(f"Interaction check failed: {e}")
        interactions_data = {}

    # Get KB drug info
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
            "high_risk": med.get("high_risk", False),
            "contraindications": info.get(
                "contraindications", []
            )[:3] if info else [],
            "monitoring": info.get("monitoring", [])[:3] if info else [],
        })

        if med.get("high_risk"):
            drug_alerts.append({
                "type": "high_risk_medication",
                "severity": "major",
                "drugs": name,
                "effect": "High-risk medication requiring careful monitoring",
                "management": f"Monitor {name} levels and adverse effects closely",
            })

    medications_str = "\n".join([
        f"{i+1}. {d['name']} {d['dosage']} {d['frequency']}"
        f"{'⚠️ HIGH-RISK' if d['high_risk'] else ''}\n"
        f"   Contraindications: "
        f"{', '.join(d['contraindications']) or 'None noted'}\n"
        f"   Monitoring: {', '.join(d['monitoring']) or 'Standard'}"
        for i, d in enumerate(drug_details)
    ]) or "No medications documented"

    interactions = interactions_data.get("interactions", [])
    interactions_str = "\n".join([
        f"- [{i.get('severity','').upper()}] "
        f"{i.get('drug_a')} + {i.get('drug_b')}: "
        f"{i.get('effect','')[:80]}"
        for i in interactions[:5]
    ]) or "No significant interactions identified"

    recommendation = interactions_data.get(
        "recommendation", "Routine monitoring recommended."
    )
    medication_safe = not any(
        a.get("severity") == "major" for a in drug_alerts
    )

    system_prompt = """You are a senior clinical pharmacist conducting 
a medication safety review. Be thorough and patient-safety focused."""

    user_prompt = f"""Clinical context: {question}
Primary Diagnosis: {primary_diagnosis}

CURRENT MEDICATIONS:
{medications_str}

DRUG INTERACTIONS IDENTIFIED:
{interactions_str}

System Recommendation: {recommendation}

Generate a clinical pharmacist assessment:

**MEDICATION SAFETY ASSESSMENT:**
[Overall safety rating and key concerns]

**CRITICAL INTERACTION ALERTS:**
[Major interactions requiring immediate action]

**HIGH-RISK MEDICATION REVIEW:**
[Detailed review of dangerous medications]

**DOSING APPROPRIATENESS:**
[Are doses appropriate for condition and patient?]

**MONITORING PLAN:**
[Specific parameters to monitor for each medication]

**PHARMACIST RECOMMENDATIONS:**
[Actionable recommendations for prescriber]

**PATIENT COUNSELING SUMMARY:**
[Key safety points for patient education]"""

    pharmacist_report = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1200,
        temperature=0.1,
    )

    logger.info(
        f"Pharmacist Agent complete | "
        f"meds={len(medications)} | "
        f"alerts={len(drug_alerts)} | "
        f"safe={medication_safe}"
    )

    return {
        "pharmacist_report": pharmacist_report,
        "drug_alerts": drug_alerts,
        "interaction_summary": interactions_str,
        "medication_safe": medication_safe,
        "agent_messages": [{
            "agent": "Pharmacist",
            "iteration": iteration,
            "medications_reviewed": len(medications),
            "alerts": len(drug_alerts),
            "medication_safe": medication_safe,
        }],
    }