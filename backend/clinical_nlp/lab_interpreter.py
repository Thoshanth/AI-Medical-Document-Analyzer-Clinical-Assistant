from backend.logger import get_logger

logger = get_logger("clinical_nlp.lab")

# Normal ranges for common lab tests
# Format: test_name_lower: (low, high, unit, clinical_meaning_if_abnormal)
NORMAL_RANGES = {
    "wbc": (4.5, 11.0, "K/uL", "Possible infection or bone marrow issue"),
    "white blood cell count": (4.5, 11.0, "K/uL", "Possible infection"),
    "rbc": (4.5, 5.9, "M/uL", "Anemia or polycythemia"),
    "red blood cell count": (4.5, 5.9, "M/uL", "Anemia or polycythemia"),
    "hemoglobin": (13.5, 17.5, "g/dL", "Anemia or polycythemia"),
    "hgb": (13.5, 17.5, "g/dL", "Anemia or polycythemia"),
    "hematocrit": (41, 53, "%", "Anemia or dehydration"),
    "hct": (41, 53, "%", "Anemia or dehydration"),
    "platelets": (150, 400, "K/uL", "Bleeding or clotting risk"),
    "glucose": (70, 100, "mg/dL", "Diabetes or hypoglycemia"),
    "fasting glucose": (70, 100, "mg/dL", "Diabetes screening"),
    "hba1c": (4.0, 5.7, "%", "Diabetes control indicator"),
    "creatinine": (0.7, 1.3, "mg/dL", "Kidney function"),
    "bun": (7, 25, "mg/dL", "Kidney/liver function"),
    "sodium": (136, 145, "meq/l", "Electrolyte imbalance"),
    "potassium": (3.5, 5.1, "meq/l", "Heart rhythm risk if abnormal"),
    "chloride": (98, 107, "meq/l", "Electrolyte balance"),
    "calcium": (8.5, 10.5, "mg/dL", "Bone/parathyroid issue"),
    "total cholesterol": (0, 200, "mg/dL", "Cardiovascular risk"),
    "cholesterol": (0, 200, "mg/dL", "Cardiovascular risk"),
    "ldl": (0, 100, "mg/dL", "Cardiovascular risk"),
    "hdl": (40, 999, "mg/dL", "Protective cholesterol"),
    "triglycerides": (0, 150, "mg/dL", "Cardiovascular/metabolic risk"),
    "tsh": (0.4, 4.0, "mIU/L", "Thyroid function"),
    "alt": (7, 56, "U/L", "Liver health"),
    "ast": (10, 40, "U/L", "Liver/heart health"),
    "alkaline phosphatase": (44, 147, "U/L", "Liver/bone disease"),
    "bilirubin": (0.2, 1.2, "mg/dL", "Liver function/jaundice"),
    "albumin": (3.4, 5.4, "g/dL", "Nutrition/liver function"),
    "uric acid": (3.5, 7.2, "mg/dL", "Gout risk"),
    "esr": (0, 20, "mm/hr", "Inflammation marker"),
    "crp": (0, 1.0, "mg/dL", "Inflammation/infection marker"),
}


def interpret_lab_values(lab_values: list[dict]) -> list[dict]:
    """
    Enriches lab values with:
    - Normal range reference
    - Clinical interpretation (normal/high/low/critical)
    - Clinical meaning of abnormal values
    - Severity assessment

    Pure rule-based — no LLM needed for standard lab interpretation.
    The normal ranges are medical standards, not LLM-generated.
    """
    logger.info(f"Interpreting {len(lab_values)} lab values")
    interpreted = []

    for lab in lab_values:
        enriched = lab.copy()
        test_name = lab.get("test_name", "").lower().strip()
        value_str = str(lab.get("value", "")).strip()

        # Try to get numeric value
        try:
            numeric_value = float(
                value_str.replace(",", "").replace(" ", "")
            )
        except ValueError:
            # Non-numeric result (positive/negative etc.)
            enriched["interpretation"] = lab.get("status", "unknown")
            enriched["normal_range"] = None
            enriched["clinical_significance"] = None
            interpreted.append(enriched)
            continue

        # Look up normal range
        range_info = None
        for range_name, (low, high, unit, meaning) in NORMAL_RANGES.items():
            if range_name in test_name or test_name in range_name:
                range_info = (low, high, unit, meaning)
                break

        if range_info:
            low, high, unit, meaning = range_info
            enriched["normal_range"] = f"{low} - {high} {unit}"

            # Determine status
            if numeric_value < low:
                deviation = (low - numeric_value) / low * 100
                if deviation > 30:
                    status = "critically_low"
                else:
                    status = "low"
            elif numeric_value > high:
                deviation = (numeric_value - high) / high * 100
                if deviation > 50:
                    status = "critically_high"
                else:
                    status = "high"
            else:
                status = "normal"
                meaning = "Within normal limits"

            enriched["interpretation"] = status
            enriched["clinical_significance"] = meaning

            if "critical" in status:
                logger.warning(
                    f"CRITICAL lab value | "
                    f"test={lab.get('test_name')} | "
                    f"value={numeric_value} | status={status}"
                )
        else:
            enriched["normal_range"] = lab.get("reference_range", "Unknown")
            enriched["interpretation"] = lab.get("status", "unknown")
            enriched["clinical_significance"] = None

        interpreted.append(enriched)

    critical_count = sum(
        1 for l in interpreted
        if "critical" in str(l.get("interpretation", ""))
    )

    logger.info(
        f"Lab interpretation complete | "
        f"total={len(interpreted)} | critical={critical_count}"
    )
    return interpreted