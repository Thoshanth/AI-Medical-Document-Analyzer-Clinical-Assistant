import json
from backend.knowledge_base.disease_db import get_disease_info
from backend.knowledge_base.drug_db import get_drug_info
from backend.knowledge_base.symptom_checker import check_symptoms
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.database.db import SessionLocal, MedicalDocument
from backend.logger import get_logger

logger = get_logger("knowledge_base.pipeline")


def enrich_with_knowledge_base(document_id: int) -> dict:
    """
    Enriches a document's clinical entities with knowledge base info.

    Requires Stage 2 analysis to have been run first.

    Flow:
    1. Load clinical entities from Stage 2
    2. Look up each diagnosis in disease database
    3. Look up each medication in drug database
    4. Run symptom checker for additional insights
    5. Return comprehensive enriched clinical picture
    """
    logger.info(f"KB enrichment | doc_id={document_id}")

    # Load Stage 2 clinical entities
    entities_data = get_clinical_entities(document_id)
    if not entities_data:
        raise ValueError(
            f"No clinical entities found for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )

    entities = entities_data.get("entities", {})

    # ── Enrich diagnoses ──────────────────────────────────────────
    enriched_diagnoses = []
    for diagnosis in entities.get("diagnoses", []):
        name = diagnosis.get("name", "")
        kb_info = get_disease_info(name)

        enriched_diagnosis = {**diagnosis}
        if kb_info:
            enriched_diagnosis["knowledge_base"] = {
                "description": kb_info.get("description", ""),
                "first_line_treatments": kb_info.get("first_line_treatments", []),
                "complications": kb_info.get("complications", []),
                "monitoring": kb_info.get("monitoring", []),
                "emergency_signs": kb_info.get("emergency_signs", []),
                "normal_lab_targets": kb_info.get("normal_lab_targets", {}),
            }
            logger.info(f"Diagnosis enriched | '{name}'")
        else:
            enriched_diagnosis["knowledge_base"] = None

        enriched_diagnoses.append(enriched_diagnosis)

    # ── Enrich medications ────────────────────────────────────────
    enriched_medications = []
    for medication in entities.get("medications", []):
        name = medication.get("name", "")
        drug_info = get_drug_info(name)

        enriched_med = {**medication}
        if drug_info:
            enriched_med["knowledge_base"] = {
                "drug_class": drug_info.get("drug_class", "Unknown"),
                "indications": drug_info.get("indications", []),
                "contraindications": drug_info.get("contraindications", []),
                "side_effects": drug_info.get("side_effects", [])[:5],
                "monitoring": drug_info.get("monitoring", []),
                "high_risk": drug_info.get("high_risk", False),
                "source": drug_info.get("source", "unknown"),
            }
            logger.info(f"Medication enriched | '{name}'")
        else:
            enriched_med["knowledge_base"] = None

        enriched_medications.append(enriched_med)

    # ── Symptom checking ──────────────────────────────────────────
    symptom_analysis = check_symptoms(
        symptoms=entities.get("symptoms", []),
        existing_diagnoses=entities.get("diagnoses", []),
    )

    # ── Build clinical summary ────────────────────────────────────
    clinical_summary = _build_clinical_summary(
        enriched_diagnoses,
        enriched_medications,
        entities,
        symptom_analysis,
    )

    logger.info(
        f"KB enrichment complete | "
        f"diagnoses={len(enriched_diagnoses)} | "
        f"medications={len(enriched_medications)} | "
        f"triage={symptom_analysis['triage_level']}"
    )

    return {
        "document_id": document_id,
        "enriched_diagnoses": enriched_diagnoses,
        "enriched_medications": enriched_medications,
        "symptom_analysis": symptom_analysis,
        "clinical_summary": clinical_summary,
        "lab_values": entities.get("lab_values", []),
        "vitals": entities.get("vitals", []),
        "procedures": entities.get("procedures", []),
    }


def _build_clinical_summary(
    diagnoses: list,
    medications: list,
    entities: dict,
    symptom_analysis: dict,
) -> dict:
    """
    Builds a structured clinical summary combining
    all knowledge base enrichments.
    """
    # Treatment gaps — conditions without medications
    treated_conditions = set()
    for med in medications:
        kb = med.get("knowledge_base", {})
        if kb:
            for indication in kb.get("indications", []):
                treated_conditions.add(indication.lower())

    treatment_gaps = []
    for diag in diagnoses:
        name = diag.get("name", "").lower()
        kb = diag.get("knowledge_base", {})
        if kb and name not in treated_conditions:
            first_line = kb.get("first_line_treatments", [])
            if first_line:
                treatment_gaps.append({
                    "condition": diag.get("name"),
                    "suggested_treatments": first_line[:3],
                })

    # Monitoring requirements
    all_monitoring = []
    for diag in diagnoses:
        kb = diag.get("knowledge_base", {})
        if kb:
            all_monitoring.extend(kb.get("monitoring", []))
    for med in medications:
        kb = med.get("knowledge_base", {})
        if kb:
            all_monitoring.extend(kb.get("monitoring", []))

    # Remove duplicates
    all_monitoring = list(set(all_monitoring))

    return {
        "triage_level": symptom_analysis.get("triage_level", "routine"),
        "severity_score": symptom_analysis.get("severity_score", 0),
        "possible_additional_conditions": symptom_analysis.get(
            "possible_conditions", []
        )[:3],
        "treatment_gaps": treatment_gaps,
        "monitoring_requirements": all_monitoring[:10],
        "total_diagnoses": len(diagnoses),
        "total_medications": len(medications),
        "high_risk_medications": [
            m.get("name") for m in medications
            if m.get("high_risk") or
            m.get("knowledge_base", {}) and
            m.get("knowledge_base", {}).get("high_risk")
        ],
    }