import json
from backend.clinical_nlp.entity_extractor import extract_clinical_entities
from backend.clinical_nlp.icd_mapper import map_to_icd10
from backend.clinical_nlp.medication_parser import normalize_medications
from backend.clinical_nlp.lab_interpreter import interpret_lab_values
from backend.database.db import SessionLocal, ClinicalEntities, MedicalDocument
from backend.logger import get_logger

logger = get_logger("clinical_nlp.pipeline")


def determine_clinical_complexity(entities: dict) -> str:
    """
    Determines clinical complexity based on number
    and types of entities found.

    low:    simple document, few entities
    medium: moderate clinical content
    high:   complex case with multiple diagnoses/medications
    """
    total = sum(len(entities.get(k, [])) for k in [
        "symptoms", "diagnoses", "medications",
        "lab_values", "vitals", "procedures"
    ])

    diagnoses = len(entities.get("diagnoses", []))
    medications = len(entities.get("medications", []))
    critical_labs = sum(
        1 for l in entities.get("lab_values", [])
        if "critical" in str(l.get("interpretation", ""))
    )
    high_risk_meds = sum(
        1 for m in entities.get("medications", [])
        if m.get("high_risk", False)
    )

    if (
        diagnoses >= 3 or
        medications >= 5 or
        critical_labs >= 1 or
        high_risk_meds >= 1 or
        total >= 15
    ):
        return "high"
    elif total >= 6 or diagnoses >= 1 or medications >= 2:
        return "medium"
    else:
        return "low"


def run_clinical_nlp(document_id: int) -> dict:
    """
    Full clinical NLP pipeline:

    1. Load document text from SQLite (Stage 1)
    2. Extract all 6 entity types (MiniMax LLM)
    3. Normalize medications (rule-based)
    4. Interpret lab values (rule-based)
    5. Map diagnoses to ICD-10 codes
    6. Assess clinical complexity
    7. Save all results to ClinicalEntities table
    8. Return comprehensive clinical report
    """
    logger.info(f"Clinical NLP pipeline | doc_id={document_id}")

    # Step 1: Load document
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(
            MedicalDocument.id == document_id
        ).first()

        if not doc:
            raise ValueError(f"Document {document_id} not found")

        text = doc.extracted_text
        doc_type = doc.document_type
        filename = doc.filename
    finally:
        db.close()

    logger.info(
        f"Document loaded | file='{filename}' | "
        f"type={doc_type} | chars={len(text)}"
    )

    # Step 2: Extract clinical entities
    entities = extract_clinical_entities(text, doc_type)

    # Step 3: Normalize medications
    if entities.get("medications"):
        entities["medications"] = normalize_medications(
            entities["medications"]
        )

    # Step 4: Interpret lab values
    if entities.get("lab_values"):
        entities["lab_values"] = interpret_lab_values(
            entities["lab_values"]
        )

    # Step 5: Map diagnoses to ICD-10
    icd10_codes = {}
    if entities.get("diagnoses"):
        icd10_codes = map_to_icd10(entities["diagnoses"])

    # Step 6: Clinical complexity
    complexity = determine_clinical_complexity(entities)

    # Step 7: Save to database
    total_entities = sum(
        len(entities.get(k, [])) for k in [
            "symptoms", "diagnoses", "medications",
            "lab_values", "vitals", "procedures"
        ]
    )

    db = SessionLocal()
    try:
        # Check if analysis already exists
        existing = db.query(ClinicalEntities).filter(
            ClinicalEntities.document_id == document_id
        ).first()

        if existing:
            # Update existing record
            existing.symptoms = json.dumps(entities.get("symptoms", []))
            existing.diagnoses = json.dumps(entities.get("diagnoses", []))
            existing.medications = json.dumps(entities.get("medications", []))
            existing.lab_values = json.dumps(entities.get("lab_values", []))
            existing.vitals = json.dumps(entities.get("vitals", []))
            existing.procedures = json.dumps(entities.get("procedures", []))
            existing.icd10_codes = json.dumps(icd10_codes)
            existing.total_entities_found = total_entities
            existing.clinical_complexity = complexity
            record_id = existing.id
            logger.info(f"Updated existing analysis | id={record_id}")
        else:
            # Create new record
            record = ClinicalEntities(
                document_id=document_id,
                symptoms=json.dumps(entities.get("symptoms", [])),
                diagnoses=json.dumps(entities.get("diagnoses", [])),
                medications=json.dumps(entities.get("medications", [])),
                lab_values=json.dumps(entities.get("lab_values", [])),
                vitals=json.dumps(entities.get("vitals", [])),
                procedures=json.dumps(entities.get("procedures", [])),
                icd10_codes=json.dumps(icd10_codes),
                total_entities_found=total_entities,
                clinical_complexity=complexity,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            record_id = record.id
            logger.info(f"Clinical entities saved | id={record_id}")

        db.commit()
    finally:
        db.close()

    logger.info(
        f"Clinical NLP complete | "
        f"entities={total_entities} | "
        f"complexity={complexity} | "
        f"icd10_mapped={len(icd10_codes)}"
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "document_type": doc_type,
        "clinical_complexity": complexity,
        "total_entities": total_entities,
        "entities": {
            "symptoms": entities.get("symptoms", []),
            "diagnoses": entities.get("diagnoses", []),
            "medications": entities.get("medications", []),
            "lab_values": entities.get("lab_values", []),
            "vitals": entities.get("vitals", []),
            "procedures": entities.get("procedures", []),
        },
        "icd10_codes": icd10_codes,
        "alerts": _generate_alerts(entities, icd10_codes),
    }


def _generate_alerts(entities: dict, icd10_codes: dict) -> list[dict]:
    """
    Generates clinical alerts based on extracted entities.
    These are important flags a clinician should see immediately.
    """
    alerts = []

    # Critical lab values
    for lab in entities.get("lab_values", []):
        if "critical" in str(lab.get("interpretation", "")):
            alerts.append({
                "type": "critical_lab",
                "severity": "high",
                "message": (
                    f"Critical lab value: {lab.get('test_name')} = "
                    f"{lab.get('value')} {lab.get('unit', '')} "
                    f"({lab.get('interpretation')})"
                ),
            })

    # High risk medications
    for med in entities.get("medications", []):
        if med.get("high_risk"):
            alerts.append({
                "type": "high_risk_medication",
                "severity": "medium",
                "message": (
                    f"High-risk medication: {med.get('name')} "
                    f"{med.get('dosage', '')} — "
                    f"requires careful monitoring"
                ),
            })

    # Severe symptoms
    for symptom in entities.get("symptoms", []):
        if symptom.get("severity") == "severe":
            alerts.append({
                "type": "severe_symptom",
                "severity": "medium",
                "message": (
                    f"Severe symptom reported: {symptom.get('name')}"
                ),
            })

    # Multiple diagnoses complexity
    if len(entities.get("diagnoses", [])) >= 3:
        alerts.append({
            "type": "complex_case",
            "severity": "low",
            "message": (
                f"Complex case: {len(entities['diagnoses'])} "
                f"diagnoses identified. Multi-disciplinary "
                f"review recommended."
            ),
        })

    logger.info(f"Generated {len(alerts)} clinical alerts")
    return alerts


def get_clinical_entities(document_id: int) -> dict | None:
    """Retrieves stored clinical entities for a document."""
    db = SessionLocal()
    try:
        record = db.query(ClinicalEntities).filter(
            ClinicalEntities.document_id == document_id
        ).first()

        if not record:
            return None

        return {
            "document_id": document_id,
            "analyzed_at": str(record.analyzed_at),
            "clinical_complexity": record.clinical_complexity,
            "total_entities": record.total_entities_found,
            "entities": {
                "symptoms": json.loads(record.symptoms),
                "diagnoses": json.loads(record.diagnoses),
                "medications": json.loads(record.medications),
                "lab_values": json.loads(record.lab_values),
                "vitals": json.loads(record.vitals),
                "procedures": json.loads(record.procedures),
            },
            "icd10_codes": json.loads(record.icd10_codes),
        }
    finally:
        db.close()