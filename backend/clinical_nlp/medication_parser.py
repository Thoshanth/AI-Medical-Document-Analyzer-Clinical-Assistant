import json
from backend.llm_client import chat_completion_json
from backend.logger import get_logger

logger = get_logger("clinical_nlp.medication")

# Frequency normalization map
FREQUENCY_MAP = {
    "qd": "once daily",
    "od": "once daily",
    "once daily": "once daily",
    "daily": "once daily",
    "bid": "twice daily",
    "bd": "twice daily",
    "twice daily": "twice daily",
    "twice a day": "twice daily",
    "2x/day": "twice daily",
    "tid": "three times daily",
    "tds": "three times daily",
    "three times daily": "three times daily",
    "3x/day": "three times daily",
    "qid": "four times daily",
    "four times daily": "four times daily",
    "4x/day": "four times daily",
    "prn": "as needed",
    "as needed": "as needed",
    "sos": "as needed",
    "weekly": "once weekly",
    "monthly": "once monthly",
    "stat": "immediately (one time)",
}

# Common drug class mapping
DRUG_CLASSES = {
    "metformin": "Biguanide (Antidiabetic)",
    "insulin": "Insulin (Antidiabetic)",
    "amoxicillin": "Penicillin Antibiotic",
    "azithromycin": "Macrolide Antibiotic",
    "ciprofloxacin": "Fluoroquinolone Antibiotic",
    "atorvastatin": "Statin (Cholesterol-lowering)",
    "metoprolol": "Beta-blocker (Antihypertensive)",
    "amlodipine": "Calcium Channel Blocker",
    "lisinopril": "ACE Inhibitor",
    "omeprazole": "Proton Pump Inhibitor",
    "pantoprazole": "Proton Pump Inhibitor",
    "aspirin": "NSAID / Antiplatelet",
    "ibuprofen": "NSAID (Anti-inflammatory)",
    "paracetamol": "Analgesic / Antipyretic",
    "acetaminophen": "Analgesic / Antipyretic",
    "warfarin": "Anticoagulant",
    "heparin": "Anticoagulant",
    "salbutamol": "Beta-2 Agonist (Bronchodilator)",
    "prednisolone": "Corticosteroid",
    "dexamethasone": "Corticosteroid",
}


def normalize_medications(medications: list[dict]) -> list[dict]:
    """
    Enriches raw medication entities with:
    - Normalized frequency (tid → three times daily)
    - Drug class identification
    - Route normalization
    - Warning flags for high-risk medications

    Does NOT call LLM — pure rule-based for speed and reliability.
    """
    logger.info(f"Normalizing {len(medications)} medications")
    normalized = []

    HIGH_RISK_DRUGS = [
        "warfarin", "heparin", "insulin", "methotrexate",
        "lithium", "digoxin", "morphine", "fentanyl",
        "vancomycin", "aminoglycosides",
    ]

    for med in medications:
        name = med.get("name", "").lower().strip()
        normalized_med = med.copy()

        # Normalize frequency
        freq = med.get("frequency", "").lower().strip()
        for abbrev, full in FREQUENCY_MAP.items():
            if abbrev in freq:
                normalized_med["frequency_normalized"] = full
                break
        else:
            normalized_med["frequency_normalized"] = freq or "not specified"

        # Add drug class
        drug_class = "Unknown"
        for drug_name, drug_cls in DRUG_CLASSES.items():
            if drug_name in name:
                drug_class = drug_cls
                break
        normalized_med["drug_class"] = drug_class

        # High risk flag
        is_high_risk = any(
            high_risk in name for high_risk in HIGH_RISK_DRUGS
        )
        normalized_med["high_risk"] = is_high_risk

        if is_high_risk:
            logger.warning(
                f"High-risk medication detected: {med.get('name')}"
            )

        normalized.append(normalized_med)

    logger.info(
        f"Medication normalization complete | "
        f"total={len(normalized)} | "
        f"high_risk={sum(1 for m in normalized if m['high_risk'])}"
    )
    return normalized