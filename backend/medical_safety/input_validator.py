import re
from backend.logger import get_logger

logger = get_logger("medical_safety.input")

# Patterns that indicate dangerous self-treatment requests
DANGEROUS_SELF_TREATMENT = [
    r"how\s+(much|many)\s+.{0,30}\s+to\s+(kill|overdose|harm)",
    r"what\s+dose\s+.{0,20}\s+(fatal|lethal|deadly)",
    r"how\s+to\s+(make|create|synthesize)\s+.{0,20}(drug|medication|poison)",
    r"(bypass|avoid|skip)\s+.{0,20}(prescription|doctor|hospital)",
    r"self.{0,10}(medicate|prescribe|diagnose)\s+without",
]

# Medical misinformation patterns
MISINFORMATION_PATTERNS = [
    r"(cure|treat)\s+cancer\s+with\s+(baking soda|lemon|turmeric)",
    r"vaccines?\s+(cause|causes)\s+(autism|harm|damage)",
    r"(don't|do not|avoid)\s+(need|take)\s+(chemo|chemotherapy|radiation)",
    r"alternative\s+(cure|treatment)\s+instead\s+of\s+(chemo|surgery)",
]

_dangerous_compiled = [
    re.compile(p, re.IGNORECASE) for p in DANGEROUS_SELF_TREATMENT
]
_misinfo_compiled = [
    re.compile(p, re.IGNORECASE) for p in MISINFORMATION_PATTERNS
]


class MedicalInputResult:
    def __init__(
        self,
        is_safe: bool,
        reason: str = "",
        threat_type: str = "",
        suggestion: str = "",
    ):
        self.is_safe = is_safe
        self.reason = reason
        self.threat_type = threat_type
        self.suggestion = suggestion

    def to_dict(self):
        return {
            "is_safe": self.is_safe,
            "reason": self.reason,
            "threat_type": self.threat_type,
            "suggestion": self.suggestion,
        }


def validate_medical_input(text: str) -> MedicalInputResult:
    """
    Validates medical input for safety.

    Checks:
    1. Input length and quality
    2. Dangerous self-treatment requests
    3. Medical misinformation promotion
    4. Prompt injection attempts
    """
    logger.debug(f"Validating medical input | chars={len(text)}")

    # Check 1: Empty or too short
    if not text or len(text.strip()) < 3:
        return MedicalInputResult(
            is_safe=False,
            reason="Input is too short or empty",
            threat_type="invalid_input",
            suggestion="Please provide a medical question or document query.",
        )

    # Check 2: Too long
    if len(text) > 8000:
        return MedicalInputResult(
            is_safe=False,
            reason="Input exceeds maximum length",
            threat_type="input_too_long",
            suggestion="Please limit your question to 8000 characters.",
        )

    # Check 3: Dangerous self-treatment
    for pattern in _dangerous_compiled:
        if pattern.search(text):
            logger.warning(
                f"Dangerous self-treatment request detected"
            )
            return MedicalInputResult(
                is_safe=False,
                reason="Request for potentially dangerous self-treatment information detected",
                threat_type="dangerous_self_treatment",
                suggestion="Please consult a qualified healthcare professional for medication guidance.",
            )

    # Check 4: Medical misinformation
    for pattern in _misinfo_compiled:
        if pattern.search(text):
            logger.warning(
                f"Medical misinformation pattern detected"
            )
            return MedicalInputResult(
                is_safe=False,
                reason="Request promotes unproven or dangerous medical misinformation",
                threat_type="medical_misinformation",
                suggestion="This platform supports evidence-based medicine only. Please consult a healthcare professional.",
            )

    # Check 5: Prompt injection
    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"you\s+are\s+now\s+a",
        r"forget\s+(your|all)\s+(instructions|rules)",
        r"jailbreak",
        r"developer\s+mode",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("Prompt injection attempt in medical system")
            return MedicalInputResult(
                is_safe=False,
                reason="Prompt injection attempt detected",
                threat_type="prompt_injection",
                suggestion="This is a medical assistant. Please ask medical questions only.",
            )

    logger.debug("Medical input validation passed")
    return MedicalInputResult(is_safe=True)