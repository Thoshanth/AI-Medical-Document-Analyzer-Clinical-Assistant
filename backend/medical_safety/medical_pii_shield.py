import re
from backend.logger import get_logger

logger = get_logger("medical_safety.pii")

# Medical-specific PII patterns (more comprehensive than Project 1)
MEDICAL_PII_PATTERNS = {
    "patient_name": re.compile(
        r"(?:patient|pt\.?|name)[\s:]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        re.IGNORECASE
    ),
    "date_of_birth": re.compile(
        r"(?:dob|date\s+of\s+birth|born\s+on)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        re.IGNORECASE
    ),
    "mrn": re.compile(
        r"(?:mrn|medical\s+record\s+(?:number|no\.?|#))[\s:]+([A-Z0-9\-]{4,15})",
        re.IGNORECASE
    ),
    "insurance_id": re.compile(
        r"(?:insurance\s+id|policy\s+(?:number|no\.?|#)|member\s+id)[\s:]+([A-Z0-9\-]{5,20})",
        re.IGNORECASE
    ),
    "ssn": re.compile(
        r"\b(\d{3}[-\s]\d{2}[-\s]\d{4})\b"
    ),
    "phone": re.compile(
        r"\b(\+?[\d\s\-\(\)]{10,15})\b"
    ),
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "aadhaar": re.compile(
        r"\b(\d{4}\s\d{4}\s\d{4})\b"
    ),
    "npi": re.compile(
        r"(?:npi|national\s+provider)[\s:]+(\d{10})",
        re.IGNORECASE
    ),
}


def detect_medical_pii(text: str) -> dict:
    """
    Detects medical PII in text.
    Returns dict of found PII types and counts.
    Does NOT return actual values — only types and counts.
    """
    found = {}
    for pii_type, pattern in MEDICAL_PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[pii_type] = len(matches)
            logger.warning(
                f"Medical PII detected | "
                f"type={pii_type} | count={len(matches)}"
            )
    return found


def redact_medical_pii(text: str) -> tuple[str, dict]:
    """
    Redacts all medical PII from text.
    Replaces with [REDACTED:TYPE] tags.

    Returns (redacted_text, redaction_summary)
    """
    redacted = text
    summary = {}

    for pii_type, pattern in MEDICAL_PII_PATTERNS.items():
        matches = pattern.findall(redacted)
        if matches:
            redacted = pattern.sub(
                f"[REDACTED:{pii_type.upper()}]", redacted
            )
            summary[pii_type] = len(matches)
            logger.info(
                f"Medical PII redacted | "
                f"type={pii_type} | count={len(matches)}"
            )

    return redacted, summary


def has_medical_pii(text: str) -> bool:
    """Quick check if text contains any medical PII."""
    return bool(detect_medical_pii(text))