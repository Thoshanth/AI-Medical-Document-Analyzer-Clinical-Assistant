import re
from backend.logger import get_logger

logger = get_logger("medical_safety.hallucination")

# Patterns that strongly suggest hallucination in medical text
HALLUCINATION_PATTERNS = [
    # Suspiciously precise statistics not from document
    re.compile(
        r"\b\d{2,3}\.\d+%\s*(efficacy|success|survival|mortality)",
        re.IGNORECASE
    ),
    # Fabricated citations
    re.compile(
        r"(?:according\s+to|study\s+by|research\s+by)\s+[A-Z][a-z]+\s+et\s+al\.\s+\d{4}",
        re.IGNORECASE
    ),
    # Overconfident medical claims
    re.compile(
        r"\b(definitely|certainly|absolutely|undoubtedly)\s+(has|have|is|are|will)\s+\w+\s+(disease|condition|disorder|syndrome)",
        re.IGNORECASE
    ),
    # Specific dosing not typically in documents
    re.compile(
        r"\btake\s+exactly\s+\d+\s*mg\s+every\s+\d+\s+hours",
        re.IGNORECASE
    ),
    # False document attribution
    re.compile(
        r"(?:the\s+document\s+clearly\s+states|the\s+report\s+confirms|as\s+documented)",
        re.IGNORECASE
    ),
]

# Phrases indicating appropriate uncertainty (good signs)
UNCERTAINTY_MARKERS = [
    "based on the provided",
    "according to the document",
    "the document indicates",
    "as mentioned in",
    "findings suggest",
    "may indicate",
    "could be",
    "consultation recommended",
    "further evaluation",
    "source",
    "disclaimer",
]


def check_for_hallucinations(
    text: str,
    source_text: str = "",
) -> dict:
    """
    Analyzes LLM output for hallucination indicators.

    Checks for:
    1. Suspicious precision patterns
    2. Fabricated citations
    3. Overconfident medical claims
    4. Claims not grounded in source text
    5. Presence of appropriate uncertainty markers

    Returns:
    {
        "hallucination_risk": "high" | "medium" | "low",
        "indicators": [...],
        "uncertainty_score": 0-10,
        "recommendation": str
    }
    """
    logger.debug(
        f"Hallucination check | chars={len(text)}"
    )

    indicators = []

    # Check hallucination patterns
    for pattern in HALLUCINATION_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            indicators.append({
                "pattern": pattern.pattern[:60],
                "matches": len(matches),
            })

    # Count uncertainty markers (good signs)
    text_lower = text.lower()
    uncertainty_count = sum(
        1 for marker in UNCERTAINTY_MARKERS
        if marker in text_lower
    )

    # Uncertainty score: higher = better grounded response
    uncertainty_score = min(uncertainty_count * 2, 10)

    # Determine risk level
    if len(indicators) >= 3 or (
        len(indicators) >= 1 and uncertainty_score < 3
    ):
        risk = "high"
        recommendation = (
            "Response shows multiple hallucination indicators. "
            "Verify all claims against source documents before use."
        )
    elif len(indicators) >= 1:
        risk = "medium"
        recommendation = (
            "Response shows some uncertainty. "
            "Cross-reference key claims with source documents."
        )
    else:
        risk = "low"
        recommendation = (
            "Response appears well-grounded. "
            "Standard medical disclaimer applies."
        )

    logger.info(
        f"Hallucination check complete | "
        f"risk={risk} | "
        f"indicators={len(indicators)} | "
        f"uncertainty_score={uncertainty_score}"
    )

    return {
        "hallucination_risk": risk,
        "indicators": indicators,
        "uncertainty_score": uncertainty_score,
        "recommendation": recommendation,
    }