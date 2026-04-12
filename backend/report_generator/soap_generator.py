from backend.llm_client import chat_completion
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.knowledge_base.kb_pipeline import enrich_with_knowledge_base
from backend.logger import get_logger
from datetime import datetime

logger = get_logger("report_generator.soap")


def generate_soap_note(document_id: int) -> dict:
    """
    Generates a structured SOAP note from a medical document.

    SOAP = Subjective, Objective, Assessment, Plan
    The most universally used clinical documentation format.

    Synthesizes:
    - Stage 2: symptoms (Subjective), labs/vitals (Objective)
    - Stage 3: KB enrichment for Assessment and Plan
    - Stage 6: Graph complications for Plan
    """
    logger.info(f"Generating SOAP note | doc_id={document_id}")

    # Load all available context
    entities_data = get_clinical_entities(document_id)
    if not entities_data:
        raise ValueError(
            f"No clinical entities for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )

    entities = entities_data.get("entities", {})

    # Try to load KB enrichment (optional)
    kb_context = ""
    try:
        kb_data = enrich_with_knowledge_base(document_id)
        summary = kb_data.get("clinical_summary", {})
        monitoring = summary.get("monitoring_requirements", [])[:5]
        gaps = summary.get("treatment_gaps", [])
        triage = summary.get("triage_level", "routine")

        kb_parts = []
        if monitoring:
            kb_parts.append(
                f"Monitoring needed: {', '.join(monitoring)}"
            )
        if gaps:
            kb_parts.append(
                f"Potential treatment gaps: "
                f"{', '.join([g.get('condition','') for g in gaps])}"
            )
        kb_parts.append(f"Triage level: {triage}")
        kb_context = "\n".join(kb_parts)
    except Exception as e:
        logger.warning(f"KB context unavailable: {e}")

    # Build entity summaries for prompt
    symptoms = entities.get("symptoms", [])
    diagnoses = entities.get("diagnoses", [])
    medications = entities.get("medications", [])
    lab_values = entities.get("lab_values", [])
    vitals = entities.get("vitals", [])
    procedures = entities.get("procedures", [])

    symptoms_str = "\n".join([
        f"- {s.get('name')} "
        f"(severity: {s.get('severity', 'unknown')}, "
        f"duration: {s.get('duration', 'not specified')})"
        for s in symptoms
    ]) or "No symptoms documented"

    diagnoses_str = "\n".join([
        f"- {d.get('name')} "
        f"[status: {d.get('status', 'unknown')}]"
        for d in diagnoses
    ]) or "No diagnoses documented"

    medications_str = "\n".join([
        f"- {m.get('name')} "
        f"{m.get('dosage', '')} "
        f"{m.get('frequency_normalized', m.get('frequency', ''))}"
        f"{'⚠️ HIGH RISK' if m.get('high_risk') else ''}"
        for m in medications
    ]) or "No medications documented"

    labs_str = "\n".join([
        f"- {l.get('test_name')}: "
        f"{l.get('value')} {l.get('unit', '')} "
        f"[{l.get('interpretation', 'unknown')}] "
        f"{'🔴 CRITICAL' if 'critical' in str(l.get('interpretation','')) else ''}"
        for l in lab_values
    ]) or "No lab values documented"

    vitals_str = "\n".join([
        f"- {v.get('name')}: {v.get('value')} "
        f"[{v.get('status', 'unknown')}]"
        for v in vitals
    ]) or "No vitals documented"

    system_prompt = """You are a senior physician generating a formal SOAP note.
Write in professional medical language.
Be precise, concise, and clinically accurate.
Every statement must be grounded in the provided data — never fabricate."""

    user_prompt = f"""Generate a complete SOAP note based on this clinical data.

=== SUBJECTIVE DATA (Patient-reported) ===
Symptoms:
{symptoms_str}

=== OBJECTIVE DATA (Measured findings) ===
Vital Signs:
{vitals_str}

Laboratory Results:
{labs_str}

Procedures Performed:
{', '.join([p.get('name','') for p in procedures]) or 'None documented'}

=== CLINICAL CONTEXT ===
Diagnoses:
{diagnoses_str}

Current Medications:
{medications_str}

Knowledge Base Context:
{kb_context or 'Not available'}

Generate a SOAP note with EXACTLY these sections:
**SUBJECTIVE:**
[Patient complaints, symptoms, history]

**OBJECTIVE:**
[Vital signs, lab results, examination findings]

**ASSESSMENT:**
[Clinical interpretation, diagnosis, severity]

**PLAN:**
[Treatment plan, monitoring, follow-up, referrals]

**CLINICAL ALERTS:**
[Any critical findings or urgent actions needed]

Write as a formal medical document. Be specific with values and dates where available."""

    soap_text = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.1,
    )

    logger.info(
        f"SOAP note generated | chars={len(soap_text)}"
    )

    return {
        "report_type": "soap_note",
        "document_id": document_id,
        "generated_at": datetime.utcnow().isoformat(),
        "report": soap_text,
        "metadata": {
            "symptoms_count": len(symptoms),
            "diagnoses_count": len(diagnoses),
            "medications_count": len(medications),
            "labs_count": len(lab_values),
        },
        "disclaimer": (
            "⚕️ This SOAP note is AI-generated for decision support "
            "only. Must be reviewed and verified by a licensed "
            "healthcare professional before clinical use."
        ),
    }