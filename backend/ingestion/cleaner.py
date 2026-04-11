import re
from backend.logger import get_logger

logger = get_logger("ingestion.cleaner")

# Medical abbreviations to expand for better LLM understanding
MEDICAL_ABBREVIATIONS = {
    r"\bbp\b": "blood pressure",
    r"\bhr\b": "heart rate",
    r"\brr\b": "respiratory rate",
    r"\btemp\b": "temperature",
    r"\bspo2\b": "oxygen saturation",
    r"\bprn\b": "as needed",
    r"\bbid\b": "twice daily",
    r"\btid\b": "three times daily",
    r"\bqid\b": "four times daily",
    r"\bqd\b": "once daily",
    r"\bpo\b": "by mouth",
    r"\biv\b": "intravenous",
    r"\bim\b": "intramuscular",
    r"\bhpi\b": "history of present illness",
    r"\bcc\b": "chief complaint",
    r"\bpe\b": "physical examination",
    r"\bdx\b": "diagnosis",
    r"\btx\b": "treatment",
    r"\bhx\b": "history",
    r"\bsx\b": "symptoms",
    r"\bnkda\b": "no known drug allergies",
    r"\ballergies\s*:\s*nkda\b": "allergies: no known drug allergies",
    r"\bwbc\b": "white blood cell count",
    r"\brbc\b": "red blood cell count",
    r"\bhgb\b": "hemoglobin",
    r"\bhct\b": "hematocrit",
    r"\bplts\b": "platelets",
    r"\bnl\b": "normal",
    r"\babn\b": "abnormal",
    r"\bpts\b": "patients",
    r"\bpt\b": "patient",
    r"\bm\.?d\.?\b": "doctor",
}


def clean_medical_text(raw_text: str, expand_abbreviations: bool = True) -> str:
    """
    Medical-specific text cleaning pipeline.

    Steps:
    1. Basic cleaning (same as Project 1)
    2. Medical abbreviation expansion (new in Project 2)
    3. Normalize medical values (lab values formatting)
    4. Remove document noise (page headers, watermarks)

    expand_abbreviations: True for clinical notes/prescriptions
                         False for research papers (abbreviations are defined)
    """
    logger.debug(f"Cleaning medical text | chars={len(raw_text)}")
    text = raw_text

    # Step 1: Basic cleaning
    text = text.replace("\x00", "")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Step 2: Remove common document noise
    # Page numbers: "Page 1 of 5", "- 1 -"
    text = re.sub(r"page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\-\s*\d+\s*\-", "", text)
    # Repeated dashes/underscores (form separators)
    text = re.sub(r"[-_=]{4,}", "---", text)
    # Remove "CONFIDENTIAL" watermarks
    text = re.sub(
        r"\b(confidential|private|draft|copy)\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Step 3: Medical abbreviation expansion
    if expand_abbreviations:
        for pattern, expansion in MEDICAL_ABBREVIATIONS.items():
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        logger.debug("Medical abbreviations expanded")

    # Step 4: Normalize lab values formatting
    # "Glucose: 126mg/dL" → "Glucose: 126 mg/dL"
    text = re.sub(r"(\d+)(mg/dl|mmol/l|g/dl|iu/l|meq/l|mm hg)",
                  r"\1 \2", text, flags=re.IGNORECASE)

    # Step 5: Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    logger.info(
        f"Cleaning complete | "
        f"input={len(raw_text)} → output={len(text)} chars"
    )
    return text


def get_word_count(text: str) -> int:
    return len(text.split())