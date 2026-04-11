import json
import simple_icd_10 as icd
from backend.llm_client import chat_completion_json
from backend.logger import get_logger

logger = get_logger("clinical_nlp.icd_mapper")

# Common diagnoses fast lookup — avoids LLM call for frequent conditions
COMMON_ICD_MAP = {
    "hypertension": "I10",
    "essential hypertension": "I10",
    "type 2 diabetes": "E11.9",
    "type 2 diabetes mellitus": "E11.9",
    "type 1 diabetes": "E10.9",
    "pneumonia": "J18.9",
    "community-acquired pneumonia": "J18.9",
    "covid-19": "U07.1",
    "covid19": "U07.1",
    "asthma": "J45.909",
    "chronic kidney disease": "N18.9",
    "ckd": "N18.9",
    "heart failure": "I50.9",
    "congestive heart failure": "I50.9",
    "atrial fibrillation": "I48.91",
    "myocardial infarction": "I21.9",
    "heart attack": "I21.9",
    "stroke": "I63.9",
    "depression": "F32.9",
    "major depressive disorder": "F32.9",
    "anxiety": "F41.9",
    "generalized anxiety disorder": "F41.1",
    "copd": "J44.9",
    "chronic obstructive pulmonary disease": "J44.9",
    "anemia": "D64.9",
    "iron deficiency anemia": "D50.9",
    "hypothyroidism": "E03.9",
    "hyperthyroidism": "E05.90",
    "obesity": "E66.9",
    "sepsis": "A41.9",
    "urinary tract infection": "N39.0",
    "uti": "N39.0",
    "appendicitis": "K37",
    "gastroesophageal reflux": "K21.9",
    "gerd": "K21.9",
    "migraine": "G43.909",
    "epilepsy": "G40.909",
    "alzheimer's disease": "G30.9",
    "parkinson's disease": "G20",
    "rheumatoid arthritis": "M06.9",
    "osteoarthritis": "M19.90",
    "osteoporosis": "M81.0",
}


def map_to_icd10(diagnoses: list[dict]) -> dict:
    """
    Maps extracted diagnoses to ICD-10 codes.

    Three-step process:
    1. Fast lookup in common diagnoses dictionary
    2. simple_icd_10 library search
    3. MiniMax LLM for complex/rare diagnoses

    Returns dict mapping diagnosis name to ICD-10 info.
    """
    if not diagnoses:
        return {}

    logger.info(f"Mapping {len(diagnoses)} diagnoses to ICD-10")
    icd_results = {}

    for diagnosis in diagnoses:
        name = diagnosis.get("name", "").lower().strip()
        if not name:
            continue

        # Step 1: Fast lookup
        if name in COMMON_ICD_MAP:
            code = COMMON_ICD_MAP[name]
            icd_results[diagnosis["name"]] = {
                "code": code,
                "description": _get_icd_description(code),
                "source": "fast_lookup",
            }
            logger.debug(f"Fast ICD lookup | '{name}' → {code}")
            continue

        # Step 2: simple_icd_10 library search
        try:
            search_results = icd.search_codes(name, 1)
            if search_results:
                code = search_results[0]
                icd_results[diagnosis["name"]] = {
                    "code": code,
                    "description": _get_icd_description(code),
                    "source": "library_search",
                }
                logger.debug(f"Library ICD lookup | '{name}' → {code}")
                continue
        except Exception as e:
            logger.debug(f"ICD library search failed for '{name}': {e}")

        # Step 3: MiniMax for complex/rare diagnoses
        logger.debug(f"Using MiniMax for ICD mapping | '{name}'")
        try:
            prompt = f"""What is the ICD-10 code for this medical diagnosis?

Diagnosis: {diagnosis['name']}

Return ONLY a JSON object:
{{"code": "ICD-10 code", "description": "official ICD-10 description"}}

If uncertain, use the closest matching code.
Return valid JSON only."""

            raw = chat_completion_json(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )

            cleaned = raw.strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]

            result = json.loads(cleaned.strip())
            icd_results[diagnosis["name"]] = {
                "code": result.get("code", "Unknown"),
                "description": result.get("description", ""),
                "source": "llm",
            }
            logger.debug(
                f"LLM ICD mapping | "
                f"'{name}' → {result.get('code')}"
            )

        except Exception as e:
            logger.warning(f"ICD mapping failed for '{name}': {e}")
            icd_results[diagnosis["name"]] = {
                "code": "Unknown",
                "description": "Could not map to ICD-10",
                "source": "failed",
            }

    logger.info(f"ICD-10 mapping complete | mapped={len(icd_results)}")
    return icd_results


def _get_icd_description(code: str) -> str:
    """Gets the official description for an ICD-10 code."""
    try:
        return icd.get_description(code)
    except Exception:
        return ""