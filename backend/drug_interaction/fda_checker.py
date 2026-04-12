import requests
from backend.logger import get_logger

logger = get_logger("drug_interaction.fda")

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"


def check_interaction_fda(
    drug_a: str,
    drug_b: str,
) -> dict | None:
    """
    Queries OpenFDA drug label API for interaction information.

    Searches drug_a's label for mentions of drug_b in the
    drug_interactions section.

    Returns interaction info if found, None otherwise.
    """
    logger.info(f"OpenFDA check | '{drug_a}' + '{drug_b}'")

    try:
        response = requests.get(
            OPENFDA_BASE,
            params={
                "search": (
                    f"openfda.generic_name:{drug_a} AND "
                    f"drug_interactions:{drug_b}"
                ),
                "limit": 1,
            },
            timeout=10,
        )

        if response.status_code != 200:
            logger.debug(f"OpenFDA returned {response.status_code}")
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            return None

        result = results[0]
        interactions_text = result.get(
            "drug_interactions", [""]
        )[0]

        if drug_b.lower() in interactions_text.lower():
            logger.info(
                f"FDA interaction found | '{drug_a}' + '{drug_b}'"
            )
            return {
                "severity": "moderate",
                "mechanism": "See FDA drug label for details",
                "effect": interactions_text[:500],
                "management": "Consult FDA drug label and healthcare provider",
                "source": "openfda",
            }

    except requests.RequestException as e:
        logger.warning(f"OpenFDA API error: {e}")

    return None