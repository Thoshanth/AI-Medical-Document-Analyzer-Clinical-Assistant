import json
from backend.llm_client import chat_completion_json
from backend.logger import get_logger

logger = get_logger("drug_interaction.llm")


def check_interaction_llm(
    drug_a: str,
    drug_b: str,
) -> dict | None:
    """
    Uses MiniMax to check drug interaction when not found
    in database or OpenFDA.

    Explicitly marks response as LLM-generated so clinicians
    know to verify independently.

    Returns interaction info or None if no interaction found.
    """
    logger.info(
        f"LLM interaction check | '{drug_a}' + '{drug_b}'"
    )

    prompt = f"""You are a clinical pharmacist. 
Analyze the drug interaction between {drug_a} and {drug_b}.

Return ONLY a JSON object:
{{
    "interaction_exists": true or false,
    "severity": "major" or "moderate" or "minor" or "none",
    "mechanism": "pharmacological mechanism",
    "effect": "clinical effect of the interaction",
    "management": "clinical management recommendation",
    "confidence": "high" or "medium" or "low"
}}

If no clinically significant interaction exists, set interaction_exists to false.
Return valid JSON only."""

    try:
        raw = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical pharmacist. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
        )

        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        result = json.loads(cleaned.strip())

        if not result.get("interaction_exists", False):
            logger.info(
                f"LLM found no interaction | "
                f"'{drug_a}' + '{drug_b}'"
            )
            return None

        interaction = {
            "severity": result.get("severity", "minor"),
            "mechanism": result.get("mechanism", ""),
            "effect": result.get("effect", ""),
            "management": result.get("management", ""),
            "source": "llm",
            "llm_confidence": result.get("confidence", "low"),
            "llm_disclaimer": (
                "This interaction was identified by AI. "
                "Please verify with official drug references "
                "or a clinical pharmacist."
            ),
        }

        logger.info(
            f"LLM interaction found | "
            f"'{drug_a}' + '{drug_b}' | "
            f"severity={interaction['severity']}"
        )
        return interaction

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"LLM interaction check failed: {e}")
        return None