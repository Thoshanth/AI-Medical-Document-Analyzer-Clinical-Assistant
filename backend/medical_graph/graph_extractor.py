import json
from backend.llm_client import chat_completion_json
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.logger import get_logger

logger = get_logger("medical_graph.extractor")


def extract_medical_graph_data(
    document_id: int,
) -> dict:
    """
    Extracts medical graph nodes and edges from a document.

    Two-step process:
    Step 1 — Convert Stage 2 clinical entities directly to graph nodes
             (fast, no LLM needed for structured entities)
    Step 2 — LLM extracts additional relationships from document text

    Returns: {entities: [...], relations: [...]}
    """
    logger.info(f"Extracting medical graph data | doc_id={document_id}")

    # Step 1: Convert Stage 2 entities to graph nodes
    entities_data = get_clinical_entities(document_id)
    entities = []
    relations = []

    if entities_data:
        stage2_entities = entities_data.get("entities", {})

        # Convert symptoms
        for symptom in stage2_entities.get("symptoms", []):
            entities.append({
                "name": symptom.get("name", "").lower(),
                "type": "symptom",
                "severity": symptom.get("severity", "unknown"),
            })

        # Convert diagnoses
        for diagnosis in stage2_entities.get("diagnoses", []):
            name = diagnosis.get("name", "").lower()
            entities.append({
                "name": name,
                "type": "disease",
                "status": diagnosis.get("status", "unknown"),
            })
            # Add symptom → disease relations
            for symptom in stage2_entities.get("symptoms", []):
                sym_name = symptom.get("name", "").lower()
                if sym_name and name:
                    relations.append({
                        "source": sym_name,
                        "relation": "INDICATES",
                        "target": name,
                        "weight": 0.7,
                    })

        # Convert medications
        for med in stage2_entities.get("medications", []):
            name = med.get("name", "").lower()
            entities.append({
                "name": name,
                "type": "drug",
                "dosage": med.get("dosage", ""),
                "high_risk": med.get("high_risk", False),
            })
            # Add disease → drug relations
            for diagnosis in stage2_entities.get("diagnoses", []):
                diag_name = diagnosis.get("name", "").lower()
                if diag_name and name:
                    relations.append({
                        "source": diag_name,
                        "relation": "TREATED_BY",
                        "target": name,
                        "weight": 0.8,
                    })

        # Convert lab values
        for lab in stage2_entities.get("lab_values", []):
            name = lab.get("test_name", "").lower()
            entities.append({
                "name": name,
                "type": "lab_test",
                "value": lab.get("value", ""),
                "status": lab.get("interpretation", "unknown"),
            })

        logger.info(
            f"Stage 2 conversion | "
            f"entities={len(entities)} | "
            f"relations={len(relations)}"
        )

    # Step 2: LLM extracts additional medical relationships
    from backend.database.db import SessionLocal, MedicalDocument
    db = SessionLocal()
    try:
        record = db.query(MedicalDocument).filter(
            MedicalDocument.id == document_id
        ).first()
        text = record.extracted_text[:3000] if record else ""
    finally:
        db.close()

    if text:
        llm_data = _llm_extract_relationships(text)
        entities.extend(llm_data.get("entities", []))
        relations.extend(llm_data.get("relations", []))

    # Deduplicate entities by name
    seen = set()
    unique_entities = []
    for e in entities:
        name = e.get("name", "").strip()
        if name and name not in seen:
            seen.add(name)
            unique_entities.append(e)

    logger.info(
        f"Extraction complete | "
        f"unique_entities={len(unique_entities)} | "
        f"relations={len(relations)}"
    )

    return {
        "entities": unique_entities,
        "relations": relations,
    }


def _llm_extract_relationships(text: str) -> dict:
    """
    Uses MiniMax to extract additional medical relationships
    not covered by Stage 2 structured extraction.
    """
    prompt = f"""Extract medical entities and clinical relationships from this text.

Text:
{text}

Return ONLY valid JSON:
{{
    "entities": [
        {{"name": "entity name lowercase", "type": "symptom|disease|drug|lab_test|procedure|anatomy|risk_factor"}}
    ],
    "relations": [
        {{"source": "source entity", "relation": "INDICATES|CAUSED_BY|TREATED_BY|DIAGNOSED_BY|MONITORED_BY|CONTRAINDICATED_IN|INTERACTS_WITH|AFFECTS|COMPLICATION_OF|RISK_FACTOR_FOR|ASSOCIATED_WITH", "target": "target entity", "weight": 0.1-1.0}}
    ]
}}

Rules:
- Use lowercase for entity names
- Only extract relationships explicitly supported by the text
- Return empty arrays if nothing found
- Valid JSON only"""

    try:
        raw = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical NLP system. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
        )

        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        result = json.loads(cleaned.strip())
        logger.info(
            f"LLM extraction | "
            f"entities={len(result.get('entities', []))} | "
            f"relations={len(result.get('relations', []))}"
        )
        return result

    except Exception as e:
        logger.warning(f"LLM graph extraction failed: {e}")
        return {"entities": [], "relations": []}