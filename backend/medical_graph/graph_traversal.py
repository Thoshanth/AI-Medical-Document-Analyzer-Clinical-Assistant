import networkx as nx
from backend.logger import get_logger

logger = get_logger("medical_graph.traversal")


def find_related_entities(
    G: nx.DiGraph,
    entity_name: str,
    relation_types: list[str] = None,
    max_hops: int = 2,
) -> dict:
    """
    Traverses the medical graph from a starting entity.
    Returns all related nodes with their relationships.

    relation_types: filter by specific relationship types
    max_hops: how many relationship hops to traverse
    """
    entity_lower = entity_name.lower().strip()

    # Find matching node (case-insensitive)
    matched_node = None
    for node in G.nodes():
        if (entity_lower in node.lower() or
                node.lower() in entity_lower):
            matched_node = node
            break

    if not matched_node:
        logger.debug(f"Entity not found in graph | '{entity_name}'")
        return {"found": False, "entity": entity_name, "related": []}

    logger.info(
        f"Traversing from '{matched_node}' | "
        f"hops={max_hops}"
    )

    related = []
    visited = {matched_node}
    frontier = {matched_node}

    for hop in range(max_hops):
        next_frontier = set()

        for node in frontier:
            # Outgoing edges
            for _, neighbor, data in G.out_edges(node, data=True):
                relation = data.get("relation", "ASSOCIATED_WITH")

                if relation_types and relation not in relation_types:
                    continue

                related.append({
                    "from": node,
                    "relation": relation,
                    "to": neighbor,
                    "node_type": G.nodes[neighbor].get("type", "unknown"),
                    "node_label": G.nodes[neighbor].get("label", neighbor),
                    "weight": data.get("weight", 0.5),
                    "hop": hop + 1,
                })

                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

            # Incoming edges
            for neighbor, _, data in G.in_edges(node, data=True):
                relation = data.get("relation", "ASSOCIATED_WITH")

                if relation_types and relation not in relation_types:
                    continue

                related.append({
                    "from": neighbor,
                    "relation": relation,
                    "to": node,
                    "node_type": G.nodes[neighbor].get("type", "unknown"),
                    "node_label": G.nodes[neighbor].get("label", neighbor),
                    "weight": data.get("weight", 0.5),
                    "hop": hop + 1,
                })

                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

        frontier = next_frontier

    # Sort by weight (most important first)
    related.sort(key=lambda x: x.get("weight", 0), reverse=True)

    logger.info(
        f"Traversal complete | "
        f"entity='{matched_node}' | "
        f"related={len(related)}"
    )

    return {
        "found": True,
        "entity": matched_node,
        "entity_type": G.nodes[matched_node].get("type", "unknown"),
        "entity_label": G.nodes[matched_node].get("label", matched_node),
        "related": related[:30],
    }


def get_treatment_pathway(
    G: nx.DiGraph,
    disease_name: str,
) -> dict:
    """
    Finds the complete treatment pathway for a disease:
    Disease → Treatments → Monitoring → Contraindications
    """
    return find_related_entities(
        G, disease_name,
        relation_types=["TREATED_BY", "MONITORED_BY",
                        "DIAGNOSED_BY", "CONTRAINDICATED_IN"],
        max_hops=2,
    )


def get_complications(
    G: nx.DiGraph,
    disease_name: str,
) -> list[dict]:
    """
    Finds all complications of a disease.
    Returns nodes connected by COMPLICATION_OF or CAUSES edges.
    """
    result = find_related_entities(
        G, disease_name,
        relation_types=["CAUSES", "COMPLICATION_OF"],
        max_hops=1,
    )
    return result.get("related", [])


def get_differential_diagnosis(
    G: nx.DiGraph,
    symptoms: list[str],
) -> list[dict]:
    """
    Given a list of symptoms, finds diseases they INDICATE.
    Higher weight = more likely diagnosis.
    """
    disease_scores = {}

    for symptom in symptoms:
        result = find_related_entities(
            G, symptom,
            relation_types=["INDICATES"],
            max_hops=1,
        )

        for item in result.get("related", []):
            if item.get("node_type") == "disease":
                disease = item["to"]
                weight = item.get("weight", 0.5)
                disease_scores[disease] = (
                    disease_scores.get(disease, 0) + weight
                )

    # Sort by cumulative score
    sorted_diseases = sorted(
        disease_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "disease": disease,
            "score": round(score, 3),
            "label": G.nodes.get(disease, {}).get("label", disease),
        }
        for disease, score in sorted_diseases[:10]
    ]


def build_patient_clinical_picture(
    G: nx.DiGraph,
    diagnoses: list[str],
    medications: list[str],
    symptoms: list[str],
) -> dict:
    """
    Builds a complete clinical picture for a patient
    by traversing the graph for all their conditions.

    Returns:
    - Complications to watch for
    - Monitoring requirements
    - Drug-condition contraindications
    - Differential diagnosis from symptoms
    """
    all_complications = []
    all_monitoring = []
    all_contraindications = []

    for diagnosis in diagnoses:
        # Complications
        complications = get_complications(G, diagnosis)
        all_complications.extend([
            {"source_disease": diagnosis, **c}
            for c in complications
        ])

        # Treatment pathway
        pathway = get_treatment_pathway(G, diagnosis)
        for item in pathway.get("related", []):
            if item.get("relation") == "MONITORED_BY":
                all_monitoring.append({
                    "for_disease": diagnosis,
                    "monitor": item.get("node_label", item["to"]),
                })

    # Drug contraindications
    for medication in medications:
        result = find_related_entities(
            G, medication,
            relation_types=["CONTRAINDICATED_IN"],
            max_hops=1,
        )
        for item in result.get("related", []):
            all_contraindications.append({
                "drug": medication,
                "contraindicated_in": item.get(
                    "node_label", item["to"]
                ),
            })

    # Differential diagnosis from symptoms
    differential = get_differential_diagnosis(G, symptoms)

    return {
        "complications_to_watch": all_complications[:10],
        "monitoring_requirements": all_monitoring[:10],
        "drug_contraindications": all_contraindications[:10],
        "differential_diagnosis": differential[:5],
    }