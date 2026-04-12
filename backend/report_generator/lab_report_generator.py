from backend.llm_client import chat_completion
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.logger import get_logger
from datetime import datetime

logger = get_logger("report_generator.lab")


def generate_lab_report(document_id: int) -> dict:
    """
    Generates a structured lab results interpretation report.

    Synthesizes Stage 2 lab values with clinical interpretations
    into a pathologist/lab medicine style report.
    """
    logger.info(
        f"Generating lab report | doc_id={document_id}"
    )

    entities_data = get_clinical_entities(document_id)
    if not entities_data:
        raise ValueError(
            f"No entities for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )

    lab_values = entities_data.get(
        "entities", {}
    ).get("lab_values", [])
    vitals = entities_data.get("entities", {}).get("vitals", [])
    diagnoses = entities_data.get(
        "entities", {}
    ).get("diagnoses", [])

    if not lab_values and not vitals:
        return {
            "report_type": "lab_interpretation",
            "document_id": document_id,
            "generated_at": datetime.utcnow().isoformat(),
            "report": "No laboratory values found in this document.",
            "critical_values": [],
            "disclaimer": "",
        }

    # Separate normal, abnormal, critical
    normal_labs = [
        l for l in lab_values
        if l.get("interpretation") == "normal"
    ]
    abnormal_labs = [
        l for l in lab_values
        if l.get("interpretation") in ["high", "low"]
    ]
    critical_labs = [
        l for l in lab_values
        if "critical" in str(l.get("interpretation", ""))
    ]

    # Format lab values for prompt
    def format_lab(l):
        interp = l.get("interpretation", "unknown")
        normal_range = l.get("normal_range", l.get("reference_range", ""))
        significance = l.get("clinical_significance", "")
        critical_flag = "🔴 CRITICAL" if "critical" in interp else ""
        return (
            f"- {l.get('test_name')}: "
            f"{l.get('value')} {l.get('unit', '')} "
            f"[{interp.upper()}] {critical_flag}\n"
            f"  Reference: {normal_range or 'Not specified'}\n"
            f"  Significance: {significance or 'Standard interpretation'}"
        )

    critical_str = "\n".join(
        [format_lab(l) for l in critical_labs]
    ) or "None"
    abnormal_str = "\n".join(
        [format_lab(l) for l in abnormal_labs]
    ) or "None"
    normal_str = "\n".join([
        f"- {l.get('test_name')}: "
        f"{l.get('value')} {l.get('unit','')} [NORMAL]"
        for l in normal_labs
    ]) or "None"

    vitals_str = "\n".join([
        f"- {v.get('name')}: {v.get('value')} [{v.get('status','')}]"
        for v in vitals
    ]) or "No vitals documented"

    diagnoses_str = ", ".join([
        d.get("name", "") for d in diagnoses
    ]) or "None documented"

    system_prompt = """You are a laboratory medicine specialist 
interpreting clinical laboratory results. Be precise, cite 
specific values, and provide clear clinical significance."""

    user_prompt = f"""Generate a structured laboratory results 
interpretation report.

KNOWN DIAGNOSES (for clinical context):
{diagnoses_str}

CRITICAL VALUES (immediate attention required):
{critical_str}

ABNORMAL VALUES:
{abnormal_str}

NORMAL VALUES:
{normal_str}

VITAL SIGNS:
{vitals_str}

Generate a lab interpretation report with EXACTLY these sections:

**LABORATORY RESULTS SUMMARY:**
[Overview: X tests performed, Y abnormal, Z critical]

**CRITICAL VALUES — IMMEDIATE ACTION REQUIRED:**
[Detail each critical value with urgency and action needed]

**ABNORMAL VALUES ANALYSIS:**
[For each abnormal value: clinical significance and interpretation]

**NORMAL VALUES:**
[Brief confirmation of normal results]

**PATTERN ANALYSIS:**
[Any patterns across multiple abnormal results]

**CLINICAL CORRELATION:**
[How lab findings relate to known diagnoses]

**RECOMMENDED FOLLOW-UP TESTING:**
[Additional tests to consider]

**LABORATORY REPORT CONCLUSION:**
[Overall summary and key action items]

Be specific with values, units, and reference ranges."""

    report_text = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.1,
    )

    logger.info(
        f"Lab report generated | chars={len(report_text)}"
    )

    return {
        "report_type": "lab_interpretation",
        "document_id": document_id,
        "generated_at": datetime.utcnow().isoformat(),
        "report": report_text,
        "critical_values": [
            {
                "test": l.get("test_name"),
                "value": l.get("value"),
                "unit": l.get("unit", ""),
                "status": l.get("interpretation"),
            }
            for l in critical_labs
        ],
        "metadata": {
            "total_labs": len(lab_values),
            "critical": len(critical_labs),
            "abnormal": len(abnormal_labs),
            "normal": len(normal_labs),
        },
        "disclaimer": (
            "⚕️ Lab interpretation is AI-generated for clinical "
            "decision support. Values must be correlated with "
            "clinical findings by a qualified physician."
        ),
    }