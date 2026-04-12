import networkx as nx
from pathlib import Path
from backend.medical_graph.graph_store import (
    get_or_build_foundation,
    save_graph,
    load_graph,
    get_document_graph_path,
    get_combined_graph_path,
)
from backend.medical_graph.graph_extractor import extract_medical_graph_data
from backend.medical_graph.graph_traversal import (
    find_related_entities,
    build_patient_clinical_picture,
)
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.llm_client import chat_completion
from backend.logger import get_logger

logger = get_logger("medical_graph.pipeline")


def build_medical_graph(document_id: int) -> dict:
    """
    Builds a combined medical graph:
    Layer 1 — Foundation knowledge graph (pre-built)
    Layer 2 — Document-specific graph (LLM extracted)
    Combined into one graph for rich traversal.
    """
    logger.info(f"Building medical graph | doc_id={document_id}")

    # Layer 1: Foundation graph
    foundation = get_or_build_foundation()
    logger.info(
        f"Foundation graph | "
        f"nodes={foundation.number_of_nodes()} | "
        f"edges={foundation.number_of_edges()}"
    )

    # Layer 2: Document-specific extraction
    doc_data = extract_medical_graph_data(document_id)

    # Build document graph
    doc_graph = nx.DiGraph()

    for entity in doc_data.get("entities", []):
        name = entity.get("name", "").strip()
        if name:
            doc_graph.add_node(
                name,
                type=entity.get("type", "unknown"),
                document_id=document_id,
                **{k: v for k, v in entity.items()
                   if k not in ["name", "type"]}
            )

    for relation in doc_data.get("relations", []):
        source = relation.get("source", "").strip()
        target = relation.get("target", "").strip()
        rel_type = relation.get("relation", "ASSOCIATED_WITH")

        if source and target:
            if source not in doc_graph:
                doc_graph.add_node(
                    source, type="unknown",
                    document_id=document_id
                )
            if target not in doc_graph:
                doc_graph.add_node(
                    target, type="unknown",
                    document_id=document_id
                )
            doc_graph.add_edge(
                source, target,
                relation=rel_type,
                weight=relation.get("weight", 0.5),
                document_id=document_id,
            )

    # Save document-specific graph
    doc_path = get_document_graph_path(document_id)
    save_graph(doc_graph, doc_path)

    # Combine: foundation + document graph
    combined = nx.compose(foundation, doc_graph)
    combined_path = get_combined_graph_path(document_id)
    save_graph(combined, combined_path)

    logger.info(
        f"Combined graph | "
        f"nodes={combined.number_of_nodes()} | "
        f"edges={combined.number_of_edges()}"
    )

    # Get node type distribution
    node_types = {}
    for _, data in combined.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    return {
        "document_id": document_id,
        "foundation_nodes": foundation.number_of_nodes(),
        "foundation_edges": foundation.number_of_edges(),
        "document_nodes": doc_graph.number_of_nodes(),
        "document_edges": doc_graph.number_of_edges(),
        "combined_nodes": combined.number_of_nodes(),
        "combined_edges": combined.number_of_edges(),
        "node_types": node_types,
        "status": "graph_built",
    }


def query_medical_graph(
    question: str,
    document_id: int,
) -> dict:
    """
    Answers a clinical question using graph traversal
    combined with medical RAG context.
    """
    logger.info(
        f"Medical graph query | "
        f"question='{question[:60]}' | "
        f"doc_id={document_id}"
    )

    # Load combined graph
    combined_path = get_combined_graph_path(document_id)
    if not combined_path.exists():
        raise ValueError(
            f"No graph for document {document_id}. "
            f"Run POST /medical-graph/build/{document_id} first."
        )

    G = load_graph(combined_path)

    # Load clinical entities for patient context
    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}

    diagnoses = [
        d.get("name", "").lower()
        for d in entities.get("diagnoses", [])
    ]
    medications = [
        m.get("name", "").lower()
        for m in entities.get("medications", [])
    ]
    symptoms = [
        s.get("name", "").lower()
        for s in entities.get("symptoms", [])
    ]

    # Build clinical picture through graph traversal
    clinical_picture = build_patient_clinical_picture(
        G, diagnoses, medications, symptoms
    )

    # Find entities mentioned in question
    question_lower = question.lower()
    graph_context_parts = []

    for node in G.nodes():
        if node.lower() in question_lower:
            result = find_related_entities(G, node, max_hops=2)
            if result.get("found"):
                related = result.get("related", [])[:5]
                if related:
                    lines = [
                        f"{r['from']} --[{r['relation']}]--> {r['to']}"
                        for r in related
                    ]
                    graph_context_parts.append(
                        f"Graph relationships for '{node}':\n" +
                        "\n".join(lines)
                    )

    graph_context = "\n\n".join(graph_context_parts)

    # Generate answer using graph context + MiniMax
    system_prompt = """You are a clinical decision support AI.
Use the medical knowledge graph data and patient clinical picture
to provide evidence-based clinical insights.

Always cite graph relationships when making clinical points.
Never make definitive diagnoses — support clinical decision-making."""

    patient_context = f"""Patient conditions: {', '.join(diagnoses) or 'None identified'}
Patient medications: {', '.join(medications) or 'None identified'}
Patient symptoms: {', '.join(symptoms) or 'None identified'}"""

    complications_str = "\n".join([
        f"- {c.get('source_disease')} → {c.get('relation')} → {c.get('to')}"
        for c in clinical_picture.get("complications_to_watch", [])[:5]
    ])

    monitoring_str = "\n".join([
        f"- Monitor {m.get('monitor')} (for {m.get('for_disease')})"
        for m in clinical_picture.get("monitoring_requirements", [])[:5]
    ])

    user_prompt = f"""Clinical Question: {question}

{patient_context}

Knowledge Graph Context:
{graph_context or 'No specific graph context for this question.'}

Complications to watch based on graph:
{complications_str or 'None identified'}

Monitoring requirements from graph:
{monitoring_str or 'None identified'}

Please answer the clinical question using this graph-enhanced context."""

    answer = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    return {
        "question": question,
        "answer": answer,
        "graph_used": True,
        "clinical_picture": clinical_picture,
        "graph_stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        },
        "disclaimer": (
            "⚕️ Graph-enhanced clinical insights are for "
            "decision support only. Always consult qualified "
            "healthcare professionals."
        ),
    }


def get_patient_summary(document_id: int) -> dict:
    """
    Generates a complete patient clinical summary
    using graph traversal across all conditions.
    """
    combined_path = get_combined_graph_path(document_id)
    if not combined_path.exists():
        raise ValueError(
            f"No graph for document {document_id}. "
            f"Build graph first."
        )

    G = load_graph(combined_path)
    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}

    diagnoses = [
        d.get("name", "").lower()
        for d in entities.get("diagnoses", [])
    ]
    medications = [
        m.get("name", "").lower()
        for m in entities.get("medications", [])
    ]
    symptoms = [
        s.get("name", "").lower()
        for s in entities.get("symptoms", [])
    ]

    clinical_picture = build_patient_clinical_picture(
        G, diagnoses, medications, symptoms
    )

    return {
        "document_id": document_id,
        "patient_conditions": diagnoses,
        "current_medications": medications,
        "reported_symptoms": symptoms,
        "clinical_picture": clinical_picture,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }


def explore_graph(document_id: int) -> dict:
    """Returns graph summary for exploration."""
    combined_path = get_combined_graph_path(document_id)
    if not combined_path.exists():
        raise ValueError(f"No graph for document {document_id}")

    G = load_graph(combined_path)

    node_types = {}
    nodes_by_type = {}
    for node, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
        if t not in nodes_by_type:
            nodes_by_type[t] = []
        nodes_by_type[t].append({
            "id": node,
            "label": data.get("label", node),
        })

    relation_types = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "UNKNOWN")
        relation_types[r] = relation_types.get(r, 0) + 1

    top_nodes = sorted(
        [(n, G.degree(n)) for n in G.nodes()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return {
        "document_id": document_id,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "relation_types": relation_types,
        "most_connected_entities": [
            {
                "entity": n,
                "connections": d,
                "label": G.nodes[n].get("label", n),
            }
            for n, d in top_nodes
        ],
        "nodes_by_type": nodes_by_type,
    }