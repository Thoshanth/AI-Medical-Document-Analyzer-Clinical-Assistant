from backend.llm_client import chat_completion
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.medical_graph.graph_store import get_combined_graph_path
from backend.medical_graph.graph_traversal import get_differential_diagnosis
from backend.medical_graph.graph_store import load_graph
from backend.logger import get_logger
from datetime import datetime

logger = get_logger("report_generator.differential")


def generate_differential_report(document_id: int) -> dict:
    """
    Generates a differential diagnosis report.

    Combines:
    - Stage 2: extracted symptoms and diagnoses
    - Stage 6: graph-based differential from symptoms
    - MiniMax: clinical reasoning for each differential

    Produces ranked list of possible diagnoses with
    supporting/opposing evidence for each.
    """
    logger.info(
        f"Generating differential diagnosis | doc_id={document_id}"
    )

    entities_data = get_clinical_entities(document_id)
    if not entities_data:
        raise ValueError(
            f"No entities for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )

    entities = entities_data.get("entities", {})
    symptoms = entities.get("symptoms", [])
    existing_diagnoses = entities.get("diagnoses", [])
    labs = entities.get("lab_values", [])

    # Graph-based differential
    graph_differential = []
    combined_path = get_combined_graph_path(document_id)
    if combined_path.exists():
        G = load_graph(combined_path)
        symptom_names = [s.get("name", "").lower() for s in symptoms]
        graph_differential = get_differential_diagnosis(G, symptom_names)

    symptoms_str = ", ".join([
        s.get("name", "") for s in symptoms
    ]) or "None documented"

    diagnoses_str = ", ".join([
        d.get("name", "") for d in existing_diagnoses
    ]) or "None documented"

    abnormal_labs = [
        l for l in labs
        if l.get("interpretation") not in ["normal", "unknown"]
    ]
    labs_str = "\n".join([
        f"- {l.get('test_name')}: {l.get('value')} "
        f"{l.get('unit','')} [{l.get('interpretation','')}]"
        for l in abnormal_labs
    ]) or "No abnormal labs"

    graph_diff_str = "\n".join([
        f"- {d.get('label', d.get('disease'))}: "
        f"score {d.get('score', 0):.2f}"
        for d in graph_differential[:5]
    ]) or "Graph differential not available"

    system_prompt = """You are a senior diagnostician generating a 
differential diagnosis report. Be systematic and evidence-based.
Only suggest diagnoses supported by the clinical data provided."""

    user_prompt = f"""Generate a differential diagnosis report.

PATIENT PRESENTATION:
Symptoms: {symptoms_str}
Known Diagnoses: {diagnoses_str}

ABNORMAL LAB VALUES:
{labs_str}

GRAPH-BASED DIFFERENTIAL (from symptom analysis):
{graph_diff_str}

Generate a structured differential diagnosis report with EXACTLY these sections:

**CLINICAL PRESENTATION SUMMARY:**
[Brief summary of key findings]

**DIFFERENTIAL DIAGNOSES:**
For each possible diagnosis provide:

1. [Diagnosis Name] - Probability: High/Medium/Low
   Supporting evidence: [from clinical data]
   Against: [what doesn't fit]
   Key distinguishing tests: [what would confirm/rule out]

2. [Continue for top 3-5 diagnoses]

**RECOMMENDED WORKUP:**
[Tests/procedures to narrow the differential]

**MOST LIKELY DIAGNOSIS:**
[Your assessment with reasoning]

**IMMEDIATE ACTIONS REQUIRED:**
[Any urgent steps needed]

Base everything strictly on the provided clinical data."""

    report_text = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.1,
    )

    logger.info(
        f"Differential report generated | chars={len(report_text)}"
    )

    return {
        "report_type": "differential_diagnosis",
        "document_id": document_id,
        "generated_at": datetime.utcnow().isoformat(),
        "report": report_text,
        "graph_differential": graph_differential[:5],
        "metadata": {
            "symptoms_analyzed": len(symptoms),
            "existing_diagnoses": len(existing_diagnoses),
            "abnormal_labs": len(abnormal_labs),
            "graph_available": combined_path.exists(),
        },
        "disclaimer": (
            "⚕️ Differential diagnosis is AI-generated for clinical "
            "decision support only. Final diagnosis must be made by "
            "a licensed physician."
        ),
    }