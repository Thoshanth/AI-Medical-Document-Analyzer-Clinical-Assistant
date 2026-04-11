import re
from backend.llm_client import chat_completion
from backend.logger import get_logger

logger = get_logger("ingestion.classifier")

KEYWORD_RULES = {
    "lab_report": [
        "blood test", "urine test", "hemoglobin", "wbc", "rbc",
        "platelet", "glucose", "cholesterol", "creatinine", "hba1c",
        "complete blood count", "cbc", "lab result", "pathology",
        "reference range", "normal range", "mg/dl", "mmol/l",
    ],
    "prescription": [
        "rx", "prescription", "prescribed by", "dispense",
        "refills", "sig:", "take one", "twice daily", "once daily",
        "tablet", "capsule", "mg", "dosage", "pharmacy",
    ],
    "clinical_note": [
        "chief complaint", "history of present illness", "hpi",
        "physical examination", "assessment and plan", "soap",
        "subjective", "objective", "diagnosis", "impression",
        "discharge summary", "progress note", "consultation",
    ],
    "research_paper": [
        "abstract", "introduction", "methodology", "results",
        "conclusion", "references", "doi:", "journal", "published",
        "clinical trial", "randomized", "placebo", "p-value",
        "confidence interval", "cohort",
    ],
    "fhir_record": [
        "resourcetype", "fhir", "hl7", "bundle", "patient",
        "observation", "condition", "medicationrequest",
    ],
}


def classify_document(text: str, filename: str) -> str:
    """
    Two-step classification:
    Step 1 — Fast keyword matching (no LLM cost)
    Step 2 — MiniMax via OpenRouter for ambiguous cases
    """
    text_lower = text.lower()[:3000]

    # JSON files are always FHIR
    if filename.endswith(".json"):
        logger.info("Classified as fhir_record (JSON file)")
        return "fhir_record"

    # Keyword scoring
    scores = {}
    for doc_type, keywords in KEYWORD_RULES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[doc_type] = score

    if scores:
        best_type = max(scores, key=lambda x: scores[x])
        if scores[best_type] >= 2:
            logger.info(
                f"Keyword classification | "
                f"type={best_type} | score={scores[best_type]}"
            )
            return best_type

    # MiniMax for ambiguous cases
    logger.info("Using MiniMax for document classification")

    prompt = f"""Classify this medical document into exactly one category.

Document preview:
{text[:1500]}

Categories:
- lab_report: blood tests, urine tests, imaging results, pathology
- prescription: medication orders, drug prescriptions
- clinical_note: doctor notes, SOAP notes, discharge summaries
- research_paper: medical studies, clinical trials, journals
- fhir_record: FHIR JSON health records
- general_medical: any other medical document

Return exactly one category name, nothing else."""

    classification = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0,
    ).strip().lower()

    valid_types = [
        "lab_report", "prescription", "clinical_note",
        "research_paper", "fhir_record", "general_medical",
    ]

    if classification not in valid_types:
        logger.warning(
            f"Invalid classification '{classification}' — "
            f"defaulting to general_medical"
        )
        classification = "general_medical"

    logger.info(f"MiniMax classification | type={classification}")
    return classification