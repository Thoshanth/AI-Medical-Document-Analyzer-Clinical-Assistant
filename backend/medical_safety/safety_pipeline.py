from fastapi import HTTPException
from backend.medical_safety.input_validator import validate_medical_input
from backend.medical_safety.emergency_detector import detect_emergency
from backend.medical_safety.medical_pii_shield import (
    detect_medical_pii,
    redact_medical_pii,
)
from backend.medical_safety.hallucination_checker import check_for_hallucinations
from backend.medical_safety.disclaimer_injector import (
    inject_disclaimer,
    select_disclaimer_type,
)
from backend.medical_safety.scope_enforcer import enforce_medical_scope
from backend.logger import get_logger

logger = get_logger("medical_safety.pipeline")


def run_medical_input_safety(question: str) -> dict:
    """
    Runs all input safety layers before question reaches LLM.

    Layers:
    1 → Input validation (dangerous requests, injection)
    2 → Emergency detection (immediate response if emergency)
    3 → Medical PII in input (block if present)
    4 → Medical scope enforcement

    Returns:
    {
        "safe": bool,
        "emergency": bool,
        "emergency_response": str or None,
        "reason": str,
        "threat_type": str
    }
    """
    logger.info(
        f"Medical input safety check | chars={len(question)}"
    )

    # Layer 1: Input validation
    validation = validate_medical_input(question)
    if not validation.is_safe:
        logger.warning(
            f"Input blocked | threat={validation.threat_type}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input blocked by medical safety system",
                "reason": validation.reason,
                "threat_type": validation.threat_type,
                "suggestion": validation.suggestion,
            }
        )

    # Layer 2: Emergency detection
    emergency = detect_emergency(question)
    if emergency["is_emergency"]:
        logger.warning(
            f"Emergency detected | type={emergency['emergency_type']}"
        )
        return {
            "safe": True,
            "emergency": True,
            "emergency_type": emergency["emergency_type"],
            "emergency_response": emergency["emergency_response"],
            "reason": "Emergency detected — immediate response provided",
            "threat_type": "medical_emergency",
        }

    # Layer 3: Medical PII in input
    pii_found = detect_medical_pii(question)
    if pii_found:
        logger.warning(
            f"Medical PII in input | types={list(pii_found.keys())}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input contains sensitive medical personal information",
                "pii_types": list(pii_found.keys()),
                "suggestion": (
                    "Please remove personal patient information "
                    "(names, DOB, MRN, insurance IDs) from your question. "
                    "Use anonymized or de-identified data."
                ),
            }
        )

    # Layer 4: Scope enforcement
    scope = enforce_medical_scope(question)
    if not scope["in_scope"]:
        logger.info(
            f"Out of scope | question='{question[:50]}'"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Question outside medical scope",
                "reason": scope["reason"],
                "suggestion": scope["suggestion"],
            }
        )

    logger.info("All medical input safety checks passed")
    return {
        "safe": True,
        "emergency": False,
        "emergency_response": None,
        "reason": "All safety checks passed",
        "threat_type": None,
    }


def run_medical_output_safety(
    answer: str,
    urgency_level: str = "routine",
) -> dict:
    """
    Runs all output safety layers on LLM response.

    Layers:
    5 → Hallucination detection
    6 → Medical PII redaction from output
    7 → Mandatory disclaimer injection

    Never blocks — always returns safe version of answer.
    """
    logger.info(
        f"Medical output safety | chars={len(answer)}"
    )

    warnings = []

    # Layer 5: Hallucination check
    hallucination = check_for_hallucinations(answer)
    if hallucination["hallucination_risk"] in ["high", "medium"]:
        warnings.append(
            f"Hallucination risk: {hallucination['hallucination_risk']} — "
            f"{hallucination['recommendation']}"
        )

    # Layer 6: PII redaction from output
    safe_answer, redactions = redact_medical_pii(answer)
    if redactions:
        warnings.append(
            f"Medical PII redacted from response: "
            f"{list(redactions.keys())}"
        )

    # Layer 7: Disclaimer injection
    disclaimer_type = select_disclaimer_type(safe_answer, urgency_level)
    final_answer = inject_disclaimer(
        safe_answer,
        disclaimer_type=disclaimer_type,
        urgency_level=urgency_level,
    )

    logger.info(
        f"Output safety complete | "
        f"hallucination_risk={hallucination['hallucination_risk']} | "
        f"pii_redacted={len(redactions)} | "
        f"disclaimer={disclaimer_type} | "
        f"warnings={len(warnings)}"
    )

    return {
        "safe_answer": final_answer,
        "original_chars": len(answer),
        "final_chars": len(final_answer),
        "hallucination_risk": hallucination["hallucination_risk"],
        "pii_redacted": redactions,
        "disclaimer_type": disclaimer_type,
        "warnings": warnings,
        "was_modified": len(redactions) > 0,
    }