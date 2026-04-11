import json
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("knowledge_base.disease")

# ── Curated Medical Disease Database ─────────────────────────────
# Each disease has: symptoms, treatments, risk_factors,
# complications, emergency_signs, icd10_code
DISEASE_DATABASE = {
    "type 2 diabetes": {
        "icd10": "E11.9",
        "full_name": "Type 2 Diabetes Mellitus",
        "description": "Chronic metabolic disorder characterized by high blood sugar due to insulin resistance",
        "symptoms": [
            "increased thirst", "frequent urination", "fatigue",
            "blurred vision", "slow wound healing", "frequent infections",
            "numbness in hands or feet", "unexplained weight loss"
        ],
        "first_line_treatments": [
            "Metformin", "lifestyle modification", "diet control",
            "regular exercise", "blood sugar monitoring"
        ],
        "second_line_treatments": [
            "Sulfonylureas", "DPP-4 inhibitors", "SGLT-2 inhibitors",
            "GLP-1 receptor agonists", "Insulin therapy"
        ],
        "risk_factors": [
            "obesity", "sedentary lifestyle", "family history",
            "age over 45", "prediabetes", "gestational diabetes history"
        ],
        "complications": [
            "diabetic nephropathy", "diabetic retinopathy",
            "diabetic neuropathy", "cardiovascular disease",
            "diabetic foot ulcers", "hyperosmolar hyperglycemic state"
        ],
        "monitoring": [
            "HbA1c every 3 months", "fasting glucose daily",
            "kidney function annually", "eye exam annually",
            "foot exam at every visit"
        ],
        "emergency_signs": [
            "blood glucose > 300 mg/dL", "ketoacidosis",
            "severe hypoglycemia", "altered consciousness"
        ],
        "normal_lab_targets": {
            "HbA1c": "< 7.0%",
            "fasting_glucose": "80-130 mg/dL",
            "postprandial_glucose": "< 180 mg/dL",
            "blood_pressure": "< 130/80 mmHg",
        }
    },
    "hypertension": {
        "icd10": "I10",
        "full_name": "Essential Hypertension",
        "description": "Persistently elevated blood pressure in arteries",
        "symptoms": [
            "headache", "dizziness", "blurred vision",
            "chest pain", "shortness of breath",
            "nosebleed", "often asymptomatic"
        ],
        "first_line_treatments": [
            "ACE inhibitors", "ARBs", "Calcium channel blockers",
            "Thiazide diuretics", "lifestyle modification"
        ],
        "second_line_treatments": [
            "Beta-blockers", "Alpha-blockers",
            "Aldosterone antagonists", "combination therapy"
        ],
        "risk_factors": [
            "age", "obesity", "high sodium diet", "sedentary lifestyle",
            "smoking", "family history", "stress", "chronic kidney disease"
        ],
        "complications": [
            "stroke", "heart attack", "heart failure",
            "kidney disease", "aneurysm", "retinopathy"
        ],
        "monitoring": [
            "blood pressure daily", "kidney function every 6 months",
            "electrolytes if on diuretics", "ECG annually"
        ],
        "emergency_signs": [
            "BP > 180/120 mmHg", "hypertensive crisis",
            "chest pain with high BP", "vision changes"
        ],
        "normal_lab_targets": {
            "blood_pressure": "< 130/80 mmHg",
            "creatinine": "0.7-1.3 mg/dL",
        }
    },
    "pneumonia": {
        "icd10": "J18.9",
        "full_name": "Pneumonia",
        "description": "Infection causing inflammation in the air sacs of one or both lungs",
        "symptoms": [
            "fever", "cough with phlegm", "chest pain",
            "shortness of breath", "fatigue", "chills",
            "confusion (elderly)", "nausea", "sweating"
        ],
        "first_line_treatments": [
            "Amoxicillin (community-acquired)",
            "Azithromycin (atypical)",
            "Levofloxacin (severe)",
            "supportive care", "oxygen therapy if needed"
        ],
        "risk_factors": [
            "age extremes", "immunocompromised", "smoking",
            "COPD", "heart failure", "hospitalization"
        ],
        "complications": [
            "sepsis", "respiratory failure", "lung abscess",
            "pleural effusion", "bacteremia"
        ],
        "monitoring": [
            "oxygen saturation", "chest X-ray",
            "CBC", "blood cultures if severe"
        ],
        "emergency_signs": [
            "SpO2 < 92%", "respiratory rate > 30/min",
            "confusion", "septic shock", "cyanosis"
        ],
        "normal_lab_targets": {
            "SpO2": "> 95%",
            "WBC": "4.5-11.0 K/uL",
            "CRP": "< 1.0 mg/dL"
        }
    },
    "anemia": {
        "icd10": "D64.9",
        "full_name": "Anemia",
        "description": "Deficiency of red blood cells or hemoglobin causing reduced oxygen delivery",
        "symptoms": [
            "fatigue", "weakness", "pale skin", "shortness of breath",
            "dizziness", "cold hands and feet", "chest pain",
            "headache", "irregular heartbeat"
        ],
        "first_line_treatments": [
            "Iron supplements (iron deficiency)",
            "Vitamin B12 injections (B12 deficiency)",
            "Folic acid (folate deficiency)",
            "treat underlying cause"
        ],
        "risk_factors": [
            "poor nutrition", "chronic disease", "blood loss",
            "pregnancy", "genetic disorders", "age"
        ],
        "complications": [
            "heart failure", "severe fatigue", "organ damage",
            "complications in pregnancy", "depression"
        ],
        "monitoring": [
            "CBC every 3 months", "iron studies",
            "vitamin B12 levels", "reticulocyte count"
        ],
        "emergency_signs": [
            "hemoglobin < 7 g/dL", "chest pain",
            "severe shortness of breath", "fainting"
        ],
        "normal_lab_targets": {
            "hemoglobin": "13.5-17.5 g/dL (male), 12-15.5 g/dL (female)",
            "ferritin": "12-300 ng/mL",
        }
    },
    "community-acquired pneumonia": {
        "icd10": "J18.9",
        "full_name": "Community-Acquired Pneumonia",
        "description": "Pneumonia acquired outside of hospital settings",
        "symptoms": [
            "productive cough", "fever", "chills",
            "pleuritic chest pain", "dyspnea", "fatigue"
        ],
        "first_line_treatments": [
            "Amoxicillin-clavulanate",
            "Azithromycin", "Doxycycline",
            "Levofloxacin (severe cases)"
        ],
        "risk_factors": [
            "age > 65", "COPD", "heart failure",
            "diabetes", "smoking", "immunosuppression"
        ],
        "complications": [
            "sepsis", "empyema", "lung abscess",
            "respiratory failure", "death (elderly)"
        ],
        "monitoring": [
            "chest X-ray at 6 weeks",
            "SpO2 monitoring", "response to antibiotics in 48-72h"
        ],
        "emergency_signs": [
            "confusion", "urea > 7 mmol/L",
            "respiratory rate >= 30", "BP < 90/60"
        ],
        "normal_lab_targets": {
            "SpO2": "> 94%",
            "WBC": "4.5-11.0 K/uL"
        }
    },
    "hypothyroidism": {
        "icd10": "E03.9",
        "full_name": "Hypothyroidism",
        "description": "Underactive thyroid producing insufficient thyroid hormone",
        "symptoms": [
            "fatigue", "weight gain", "cold intolerance",
            "constipation", "dry skin", "hair loss",
            "slow heart rate", "depression", "memory problems"
        ],
        "first_line_treatments": [
            "Levothyroxine (T4 replacement)",
            "dose adjusted by TSH levels"
        ],
        "risk_factors": [
            "female sex", "age > 60", "autoimmune disease",
            "thyroid surgery", "radiation therapy", "family history"
        ],
        "complications": [
            "myxedema coma", "heart disease",
            "infertility", "birth defects", "peripheral neuropathy"
        ],
        "monitoring": [
            "TSH every 6-12 months",
            "T4 levels", "cholesterol levels"
        ],
        "emergency_signs": [
            "myxedema coma", "severe bradycardia",
            "hypothermia", "altered consciousness"
        ],
        "normal_lab_targets": {
            "TSH": "0.4-4.0 mIU/L",
            "Free T4": "0.8-1.8 ng/dL"
        }
    },
}


def get_disease_info(disease_name: str) -> dict | None:
    """
    Looks up disease information from the knowledge base.
    Case-insensitive, partial match supported.
    """
    name_lower = disease_name.lower().strip()

    # Exact match first
    if name_lower in DISEASE_DATABASE:
        logger.info(f"Disease found (exact) | '{disease_name}'")
        return DISEASE_DATABASE[name_lower]

    # Partial match
    for db_name, info in DISEASE_DATABASE.items():
        if db_name in name_lower or name_lower in db_name:
            logger.info(f"Disease found (partial) | '{disease_name}' → '{db_name}'")
            return info

    logger.debug(f"Disease not in local DB | '{disease_name}'")
    return None


def get_all_diseases() -> list[str]:
    """Returns list of all disease names in the database."""
    return list(DISEASE_DATABASE.keys())


def search_diseases_by_symptom(symptom: str) -> list[str]:
    """
    Finds diseases that commonly present with a given symptom.
    Used by the symptom checker.
    """
    symptom_lower = symptom.lower()
    matching_diseases = []

    for disease_name, info in DISEASE_DATABASE.items():
        if any(symptom_lower in s for s in info.get("symptoms", [])):
            matching_diseases.append(disease_name)

    logger.debug(
        f"Symptom search | '{symptom}' → {len(matching_diseases)} diseases"
    )
    return matching_diseases