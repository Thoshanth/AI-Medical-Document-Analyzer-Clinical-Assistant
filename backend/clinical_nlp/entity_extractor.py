import json
from backend.llm_client import chat_completion_json
from backend.logger import get_logger

logger = get_logger("clinical_nlp.extractor")


def extract_clinical_entities(text: str, doc_type: str) -> dict:
    """
    Uses MiniMax to extract all 6 clinical entity types
    from medical document text.

    Returns structured JSON with all entities found.
    Different prompts for different document types
    for better accuracy.
    """
    logger.info(
        f"Extracting clinical entities | "
        f"doc_type={doc_type} | chars={len(text)}"
    )

    # Use first 4000 chars — enough context without hitting token limits
    text_sample = text[:4000]

    prompt = f"""You are a clinical NLP system. Extract all medical entities from this {doc_type}.

Medical Document:
{text_sample}

Extract and return ONLY a valid JSON object with this exact structure:
{{
    "symptoms": [
        {{"name": "symptom name", "severity": "mild|moderate|severe|unknown", "duration": "duration if mentioned or null"}}
    ],
    "diagnoses": [
        {{"name": "diagnosis name", "status": "confirmed|suspected|ruled_out", "notes": "any additional context or null"}}
    ],
    "medications": [
        {{"name": "drug name", "dosage": "dose and unit or null", "frequency": "how often or null", "duration": "how long or null", "route": "oral|iv|topical|other|null"}}
    ],
    "lab_values": [
        {{"test_name": "test name", "value": "numeric value", "unit": "unit of measurement or null", "status": "normal|high|low|critical|unknown", "reference_range": "range if mentioned or null"}}
    ],
    "vitals": [
        {{"name": "vital sign name", "value": "value with unit", "status": "normal|abnormal|critical|unknown"}}
    ],
    "procedures": [
        {{"name": "procedure name", "status": "performed|planned|recommended|null"}}
    ]
}}

Rules:
- Only extract information explicitly present in the text
- Return empty arrays [] if a category has no entities
- Keep entity names concise and standardized
- Return valid JSON only, no explanation or markdown
- For medications normalize frequency: qd=once daily, bid=twice daily, tid=three times daily, qid=four times daily"""

    try:
        raw = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical NLP system. Always return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
        )

        # Clean markdown if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        entities = json.loads(cleaned)

        # Ensure all keys exist
        default_keys = [
            "symptoms", "diagnoses", "medications",
            "lab_values", "vitals", "procedures"
        ]
        for key in default_keys:
            if key not in entities:
                entities[key] = []

        # Count total entities
        total = sum(len(entities[k]) for k in default_keys)

        logger.info(
            f"Entities extracted | total={total} | "
            f"symptoms={len(entities['symptoms'])} | "
            f"diagnoses={len(entities['diagnoses'])} | "
            f"medications={len(entities['medications'])} | "
            f"labs={len(entities['lab_values'])} | "
            f"vitals={len(entities['vitals'])} | "
            f"procedures={len(entities['procedures'])}"
        )

        return entities

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e} | raw='{raw[:200]}'")
        return {
            "symptoms": [], "diagnoses": [], "medications": [],
            "lab_values": [], "vitals": [], "procedures": [],
        }
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}", exc_info=True)
        return {
            "symptoms": [], "diagnoses": [], "medications": [],
            "lab_values": [], "vitals": [], "procedures": [],
        }