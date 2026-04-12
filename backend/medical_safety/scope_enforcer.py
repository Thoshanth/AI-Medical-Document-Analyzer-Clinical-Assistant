import re
from backend.llm_client import chat_completion
from backend.logger import get_logger

logger = get_logger("medical_safety.scope")

# Topics clearly within medical scope — fast pass
IN_SCOPE_KEYWORDS = [
    "symptom", "diagnosis", "medication", "drug", "treatment",
    "lab", "test", "result", "blood", "disease", "condition",
    "patient", "doctor", "clinical", "medical", "health",
    "prescription", "dosage", "hospital", "surgery", "procedure",
    "vital", "heart", "lung", "kidney", "liver", "cancer",
    "diabetes", "hypertension", "infection", "pain", "fever",
    "report", "document", "fhir", "icd", "diagnosis",
]

# Topics clearly outside medical scope — fast block
OUT_OF_SCOPE_KEYWORDS = [
    "stock market", "investment", "cryptocurrency", "bitcoin",
    "relationship advice", "dating", "love letter",
    "write a song", "poem", "creative writing",
    "legal advice", "lawsuit", "court",
    "cooking recipe", "food recipe",
    "travel itinerary", "vacation",
    "sports score", "game result",
    "homework", "essay writing",
]


def enforce_medical_scope(question: str) -> dict:
    """
    Enforces that questions are within medical scope.

    Two-step check:
    Step 1 — Fast keyword check (no LLM)
    Step 2 — LLM scope check for ambiguous cases

    Returns:
    {
        "in_scope": bool,
        "reason": str,
        "suggestion": str
    }
    """
    question_lower = question.lower()

    # Fast pass — clearly medical
    for keyword in IN_SCOPE_KEYWORDS:
        if keyword in question_lower:
            logger.debug(f"In-scope (keyword match) | '{keyword}'")
            return {
                "in_scope": True,
                "reason": "Medical topic detected",
                "suggestion": "",
            }

    # Fast block — clearly out of scope
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in question_lower:
            logger.info(f"Out of scope | keyword='{keyword}'")
            return {
                "in_scope": False,
                "reason": f"Question appears unrelated to medical topics",
                "suggestion": (
                    "This platform specializes in medical document analysis. "
                    "Please ask questions about medical documents, symptoms, "
                    "diagnoses, medications, or clinical findings."
                ),
            }

    # LLM check for ambiguous questions
    logger.debug("Using LLM for scope check")

    prompt = f"""Is this question related to medicine, healthcare, or medical documents?

Question: "{question}"

Answer with exactly one word: MEDICAL or NOT_MEDICAL"""

    try:
        decision = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        ).strip().upper()

        in_scope = "MEDICAL" in decision

        logger.info(
            f"LLM scope check | "
            f"question='{question[:40]}' | in_scope={in_scope}"
        )

        return {
            "in_scope": in_scope,
            "reason": (
                "Medical question confirmed" if in_scope
                else "Question outside medical scope"
            ),
            "suggestion": (
                "" if in_scope
                else "Please ask questions related to medical documents, symptoms, diagnoses, or treatments."
            ),
        }

    except Exception as e:
        logger.warning(f"Scope check LLM failed: {e} — defaulting to in-scope")
        return {
            "in_scope": True,
            "reason": "Scope check unavailable — defaulting to permitted",
            "suggestion": "",
        }