import itertools
import json
from backend.drug_interaction.interaction_db import (
    check_pair_in_db,
    get_all_interactions_for_drug,
)
from backend.drug_interaction.fda_checker import check_interaction_fda
from backend.drug_interaction.llm_checker import check_interaction_llm
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.logger import get_logger

logger = get_logger("drug_interaction.pipeline")

SEVERITY_ORDER = {"major": 3, "moderate": 2, "minor": 1, "none": 0}


def check_drug_pair(
    drug_a: str,
    drug_b: str,
) -> dict:
    """
    Checks interaction between two drugs using
    three-source cascade:

    Source 1 → Curated database (instant, most reliable)
    Source 2 → OpenFDA API (authoritative, live data)
    Source 3 → MiniMax LLM (broad knowledge, marked as AI)

    Returns first interaction found — stops at most severe source.
    """
    logger.info(f"Checking pair | '{drug_a}' + '{drug_b}'")

    # Source 1: Curated database
    interaction = check_pair_in_db(drug_a, drug_b)
    if interaction:
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_found": True,
            "source": "curated_database",
            **interaction,
        }

    # Source 2: OpenFDA
    interaction = check_interaction_fda(drug_a, drug_b)
    if interaction:
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_found": True,
            **interaction,
        }

    # Source 3: MiniMax LLM
    interaction = check_interaction_llm(drug_a, drug_b)
    if interaction:
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_found": True,
            **interaction,
        }

    # No interaction found
    logger.info(f"No interaction found | '{drug_a}' + '{drug_b}'")
    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "interaction_found": False,
        "severity": "none",
        "source": "all_sources_checked",
    }


def check_all_medications(document_id: int) -> dict:
    """
    Checks all medication pairs from a document.

    Flow:
    1. Load medications from Stage 2 clinical entities
    2. Generate all unique pairs (combinations not permutations)
    3. Check each pair through three-source cascade
    4. Sort by severity (major first)
    5. Generate pharmacist-style report

    Returns comprehensive interaction report.
    """
    logger.info(f"Checking all medications | doc_id={document_id}")

    # Load Stage 2 clinical entities
    entities_data = get_clinical_entities(document_id)
    if not entities_data:
        raise ValueError(
            f"No clinical entities for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )

    medications = entities_data.get("entities", {}).get("medications", [])

    if not medications:
        return {
            "document_id": document_id,
            "medications_found": [],
            "total_medications": 0,
            "pairs_checked": 0,
            "interactions_found": 0,
            "interactions": [],
            "summary": "No medications found in document.",
            "recommendation": "No drug interaction check needed.",
        }

    # Extract medication names
    med_names = [
        m.get("name", "") for m in medications
        if m.get("name")
    ]

    logger.info(
        f"Medications to check | count={len(med_names)} | "
        f"names={med_names}"
    )

    # Generate all unique pairs
    pairs = list(itertools.combinations(med_names, 2))
    logger.info(f"Drug pairs to check | count={len(pairs)}")

    # Check each pair
    all_interactions = []
    interactions_found = []

    for drug_a, drug_b in pairs:
        result = check_drug_pair(drug_a, drug_b)
        all_interactions.append(result)

        if result.get("interaction_found"):
            interactions_found.append(result)

    # Sort by severity (major first)
    interactions_found.sort(
        key=lambda x: SEVERITY_ORDER.get(x.get("severity", "none"), 0),
        reverse=True,
    )

    # Count by severity
    major_count = sum(
        1 for i in interactions_found
        if i.get("severity") == "major"
    )
    moderate_count = sum(
        1 for i in interactions_found
        if i.get("severity") == "moderate"
    )
    minor_count = sum(
        1 for i in interactions_found
        if i.get("severity") == "minor"
    )

    # Generate clinical recommendation
    recommendation = _generate_recommendation(
        major_count,
        moderate_count,
        minor_count,
        medications,
    )

    logger.info(
        f"Interaction check complete | "
        f"pairs={len(pairs)} | found={len(interactions_found)} | "
        f"major={major_count} | moderate={moderate_count} | "
        f"minor={minor_count}"
    )

    return {
        "document_id": document_id,
        "medications_found": [
            {
                "name": m.get("name"),
                "dosage": m.get("dosage"),
                "frequency": m.get("frequency_normalized",
                                   m.get("frequency")),
                "high_risk": m.get("high_risk", False),
            }
            for m in medications
        ],
        "total_medications": len(med_names),
        "pairs_checked": len(pairs),
        "interactions_found": len(interactions_found),
        "severity_summary": {
            "major": major_count,
            "moderate": moderate_count,
            "minor": minor_count,
        },
        "interactions": interactions_found,
        "all_pairs_checked": all_interactions,
        "recommendation": recommendation,
        "disclaimer": (
            "⚕️ This drug interaction check is for informational "
            "purposes only. Always consult a qualified pharmacist "
            "or physician before making medication decisions."
        ),
    }


def _generate_recommendation(
    major: int,
    moderate: int,
    minor: int,
    medications: list,
) -> str:
    """
    Generates a clinical recommendation based on
    interaction severity counts.
    """
    high_risk_count = sum(
        1 for m in medications if m.get("high_risk")
    )

    if major > 0:
        return (
            f"🔴 URGENT: {major} major drug interaction(s) identified. "
            f"Immediate pharmacist and physician review required. "
            f"Do not administer until interactions are resolved."
        )
    elif moderate > 0 or high_risk_count > 0:
        return (
            f"🟡 CAUTION: {moderate} moderate interaction(s) found. "
            f"Pharmacist review recommended before dispensing. "
            f"Monitor patient closely if medications are continued."
        )
    elif minor > 0:
        return (
            f"🟢 MINOR: {minor} minor interaction(s) noted. "
            f"No immediate action required but document in patient record. "
            f"Routine monitoring recommended."
        )
    else:
        return (
            "✅ No significant drug interactions identified "
            "in checked medication combinations. "
            "Continue routine monitoring."
        )