import json
from backend.llm_client import chat_completion_json, chat_completion
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.ingestion.medical_metadata import EMERGENCY_KEYWORDS
from backend.logger import get_logger

logger = get_logger("clinical_agents.triage")


def triage_agent(state: dict) -> dict:
    """
    Agent 1 — Triage Agent

    First agent to run. Determines clinical urgency.
    Can immediately flag emergency cases for fast-tracking.

    Reads:  document_id, patient_question
    Writes: triage_report, urgency_level, triage_alerts, is_emergency
    """
    document_id = state["document_id"]
    question = state["patient_question"]
    iteration = state.get("iterations", 1)

    logger.info(
        f"Triage Agent | doc_id={document_id} | iter={iteration}"
    )

    # Load clinical entities from Stage 2
    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}

    symptoms = entities.get("symptoms", [])
    diagnoses = entities.get("diagnoses", [])
    lab_values = entities.get("lab_values", [])
    vitals = entities.get("vitals", [])

    # Fast emergency detection from critical labs
    critical_labs = [
        l for l in lab_values
        if "critical" in str(l.get("interpretation", ""))
    ]

    severe_symptoms = [
        s for s in symptoms
        if s.get("severity") == "severe"
    ]

    # Check for emergency keywords in symptoms/diagnoses
    all_clinical_text = " ".join([
        s.get("name", "") for s in symptoms
    ] + [
        d.get("name", "") for d in diagnoses
    ]).lower()

    emergency_indicators = [
        kw for kw in EMERGENCY_KEYWORDS
        if kw in all_clinical_text
    ]

    # Build context for LLM
    symptoms_str = ", ".join([s.get("name", "") for s in symptoms]) or "None"
    diagnoses_str = ", ".join([d.get("name", "") for d in diagnoses]) or "None"

    critical_str = "\n".join([
        f"- {l.get('test_name')}: {l.get('value')} "
        f"[CRITICAL {l.get('interpretation','')}]"
        for l in critical_labs
    ]) or "None"

    vitals_str = "\n".join([
        f"- {v.get('name')}: {v.get('value')} [{v.get('status','')}]"
        for v in vitals
    ]) or "None"

    prompt = f"""You are a clinical triage nurse assessing a patient case.

Clinical Question: {question}

Patient Symptoms: {symptoms_str}
Known Diagnoses: {diagnoses_str}
Critical Lab Values: {critical_str}
Vital Signs: {vitals_str}
Emergency Indicators Found: {', '.join(emergency_indicators) or 'None'}

Assess urgency and return ONLY valid JSON:
{{
    "urgency_level": "emergency" | "urgent" | "routine",
    "is_emergency": true | false,
    "primary_concern": "main clinical concern in one sentence",
    "triage_alerts": ["alert 1", "alert 2"],
    "recommended_immediate_actions": ["action 1", "action 2"],
    "triage_summary": "2-3 sentence clinical triage summary"
}}

Emergency = immediate life threat requiring instant intervention
Urgent = needs attention within hours
Routine = can be addressed in normal clinical workflow"""

    try:
        raw = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical triage specialist. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
        )

        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        result = json.loads(cleaned.strip())

        urgency = result.get("urgency_level", "routine")
        is_emergency = (
            result.get("is_emergency", False) or
            len(critical_labs) >= 2 or
            len(emergency_indicators) >= 2
        )

        triage_report = f"""=== TRIAGE ASSESSMENT ===
Urgency Level: {urgency.upper()}
Primary Concern: {result.get('primary_concern', '')}

Triage Alerts:
{chr(10).join(f"- {a}" for a in result.get('triage_alerts', []))}

Immediate Actions Required:
{chr(10).join(f"- {a}" for a in result.get('recommended_immediate_actions', []))}

Summary: {result.get('triage_summary', '')}"""

        logger.info(
            f"Triage complete | urgency={urgency} | "
            f"emergency={is_emergency} | "
            f"alerts={len(result.get('triage_alerts', []))}"
        )

        if is_emergency:
            logger.warning(
                f"EMERGENCY CASE DETECTED | doc_id={document_id}"
            )

        return {
            "triage_report": triage_report,
            "urgency_level": urgency,
            "triage_alerts": result.get("triage_alerts", []),
            "is_emergency": is_emergency,
            "agent_messages": [{
                "agent": "Triage",
                "iteration": iteration,
                "urgency": urgency,
                "is_emergency": is_emergency,
                "summary": result.get("triage_summary", "")[:100],
            }],
        }

    except Exception as e:
        logger.error(f"Triage agent failed: {e}", exc_info=True)
        return {
            "triage_report": "Triage assessment failed — defaulting to urgent.",
            "urgency_level": "urgent",
            "triage_alerts": ["Triage system error — manual review required"],
            "is_emergency": False,
            "agent_messages": [{
                "agent": "Triage",
                "iteration": iteration,
                "error": str(e),
            }],
        }