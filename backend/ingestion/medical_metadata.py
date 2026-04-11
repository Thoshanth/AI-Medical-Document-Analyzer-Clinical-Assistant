import re
from backend.logger import get_logger

logger = get_logger("ingestion.metadata")

# Urgency keywords — critical for medical prioritization
EMERGENCY_KEYWORDS = [
    "emergency", "critical", "stat", "urgent", "immediately",
    "life-threatening", "severe", "acute", "crisis", "code blue",
    "chest pain", "stroke", "heart attack", "anaphylaxis",
    "hemorrhage", "respiratory failure", "cardiac arrest",
]

URGENT_KEYWORDS = [
    "urgent", "expedite", "priority", "as soon as possible",
    "asap", "within 24 hours", "elevated", "abnormal", "high",
    "concerning", "follow up immediately",
]

# Medical PII patterns
MEDICAL_PII_PATTERNS = {
    "patient_name": re.compile(
        r"patient\s*(?:name)?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        re.IGNORECASE
    ),
    "date_of_birth": re.compile(
        r"(?:dob|date\s+of\s+birth|born)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        re.IGNORECASE
    ),
    "mrn": re.compile(
        r"(?:mrn|medical\s+record\s+(?:number|no|#))[\s:]+([A-Z0-9\-]+)",
        re.IGNORECASE
    ),
    "insurance_id": re.compile(
        r"(?:insurance\s+id|policy\s+(?:number|no|#)|member\s+id)[\s:]+([A-Z0-9\-]+)",
        re.IGNORECASE
    ),
    "npi": re.compile(
        r"(?:npi|national\s+provider\s+identifier)[\s:]+(\d{10})",
        re.IGNORECASE
    ),
}

# Medical facility patterns
FACILITY_PATTERNS = re.compile(
    r"(?:hospital|clinic|medical\s+center|health\s+system|"
    r"institute|healthcare|infirmary)\s*(?:[:\-]\s*)?([A-Z][^\n,]{3,50})",
    re.IGNORECASE
)

# Date patterns for document date
DATE_PATTERNS = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},?\s+\d{4})\b"
)


def extract_medical_metadata(text: str, doc_type: str) -> dict:
    """
    Extracts rich medical metadata from document text.

    Returns:
    {
        "urgency_level": "emergency|urgent|routine",
        "has_pii": True/False,
        "pii_types_found": ["patient_name", "mrn"],
        "medical_facility": "City General Hospital",
        "document_date": "2026-04-01",
        "urgency_indicators": ["critical", "stat"],
        "document_type": "lab_report",
    }
    """
    logger.debug(f"Extracting medical metadata | doc_type={doc_type}")
    text_lower = text.lower()

    # Urgency detection
    emergency_found = [
        kw for kw in EMERGENCY_KEYWORDS if kw in text_lower
    ]
    urgent_found = [
        kw for kw in URGENT_KEYWORDS if kw in text_lower
    ]

    if emergency_found:
        urgency = "emergency"
    elif urgent_found:
        urgency = "urgent"
    else:
        urgency = "routine"

    # PII detection
    pii_found = []
    for pii_type, pattern in MEDICAL_PII_PATTERNS.items():
        if pattern.search(text):
            pii_found.append(pii_type)

    # Medical facility
    facility_match = FACILITY_PATTERNS.search(text)
    facility = facility_match.group(0).strip() if facility_match else None

    # Document date
    date_matches = DATE_PATTERNS.findall(text)
    doc_date = date_matches[0] if date_matches else None

    metadata = {
        "urgency_level": urgency,
        "has_pii": len(pii_found) > 0,
        "pii_types_found": pii_found,
        "medical_facility": facility,
        "document_date": doc_date,
        "urgency_indicators": emergency_found + urgent_found,
        "document_type": doc_type,
    }

    logger.info(
        f"Metadata extracted | urgency={urgency} | "
        f"pii={len(pii_found)} types | "
        f"facility={facility}"
    )

    if urgency == "emergency":
        logger.warning(
            f"EMERGENCY DOCUMENT DETECTED | "
            f"indicators={emergency_found}"
        )

    return metadata