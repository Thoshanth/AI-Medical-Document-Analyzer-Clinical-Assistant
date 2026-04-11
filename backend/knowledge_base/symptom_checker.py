from backend.knowledge_base.disease_db import (
    search_diseases_by_symptom,
    get_disease_info,
)
from backend.llm_client import chat_completion
from backend.logger import get_logger

logger = get_logger("knowledge_base.symptom_checker")

# Symptom severity weights
# Higher weight = more clinically significant symptom
SYMPTOM_WEIGHTS = {
    "chest pain": 10,
    "shortness of breath": 9,
    "altered consciousness": 10,
    "seizure": 10,
    "severe bleeding": 10,
    "fever": 5,
    "cough": 3,
    "fatigue": 2,
    "headache": 4,
    "nausea": 2,
    "dizziness": 4,
    "palpitations": 6,
    "edema": 5,
    "jaundice": 7,
    "hematuria": 7,
}


def check_symptoms(
    symptoms: list[dict],
    existing_diagnoses: list[dict],
) -> dict:
    """
    Cross-references extracted symptoms against the disease database
    to find possible conditions not yet diagnosed.

    Also scores symptom severity to help with triage.

    Returns:
    {
        "possible_conditions": [...],
        "severity_score": 0-100,
        "triage_level": "emergency|urgent|routine",
        "missing_diagnoses": [...],
    }
    """
    logger.info(
        f"Symptom checker | symptoms={len(symptoms)} | "
        f"existing_diagnoses={len(existing_diagnoses)}"
    )

    if not symptoms:
        return {
            "possible_conditions": [],
            "severity_score": 0,
            "triage_level": "routine",
            "missing_diagnoses": [],
        }

    # Get existing diagnosis names for comparison
    existing_names = {
        d.get("name", "").lower() for d in existing_diagnoses
    }

    # Find possible conditions for each symptom
    condition_scores = {}
    for symptom in symptoms:
        symptom_name = symptom.get("name", "").lower()
        severity = symptom.get("severity", "unknown")

        # Search disease database
        matching_diseases = search_diseases_by_symptom(symptom_name)
        for disease in matching_diseases:
            if disease not in condition_scores:
                condition_scores[disease] = 0
            condition_scores[disease] += 1

            # Extra weight for severe symptoms
            if severity == "severe":
                condition_scores[disease] += 2

    # Sort by score
    sorted_conditions = sorted(
        condition_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Find possible conditions not already diagnosed
    possible = []
    missing_diagnoses = []

    for condition, score in sorted_conditions[:5]:
        info = get_disease_info(condition)
        condition_entry = {
            "condition": condition,
            "match_score": score,
            "icd10": info.get("icd10", "Unknown") if info else "Unknown",
            "already_diagnosed": condition in existing_names,
        }
        possible.append(condition_entry)

        if condition not in existing_names and score >= 2:
            missing_diagnoses.append(condition)

    # Calculate overall severity score (0-100)
    severity_score = 0
    for symptom in symptoms:
        name = symptom.get("name", "").lower()
        sev = symptom.get("severity", "mild")

        # Weight from severity map
        weight = 0
        for key, w in SYMPTOM_WEIGHTS.items():
            if key in name:
                weight = w
                break
        else:
            weight = 2  # default weight

        # Multiply by severity modifier
        modifiers = {
            "severe": 3, "moderate": 2, "mild": 1, "unknown": 1
        }
        severity_score += weight * modifiers.get(sev, 1)

    # Normalize to 0-100
    severity_score = min(severity_score, 100)

    # Determine triage level
    if severity_score >= 60:
        triage = "emergency"
    elif severity_score >= 30:
        triage = "urgent"
    else:
        triage = "routine"

    logger.info(
        f"Symptom check complete | "
        f"conditions={len(possible)} | "
        f"severity={severity_score} | triage={triage}"
    )

    return {
        "possible_conditions": possible,
        "severity_score": severity_score,
        "triage_level": triage,
        "missing_diagnoses": missing_diagnoses,
    }