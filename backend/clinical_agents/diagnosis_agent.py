import json
from backend.llm_client import chat_completion
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.medical_graph.graph_store import get_combined_graph_path, load_graph
from backend.medical_graph.graph_traversal import (
    get_differential_diagnosis,
    build_patient_clinical_picture,
)
from backend.logger import get_logger

logger = get_logger("clinical_agents.diagnosis")


def diagnosis_agent(state: dict) -> dict:
    """
    Agent 2 — Diagnosis Agent

    Clinical diagnostician. Analyzes all clinical findings
    and uses the medical knowledge graph for differential diagnosis.

    Reads:  document_id, triage_report, urgency_level
    Writes: diagnosis_report, primary_diagnosis, differential_diagnoses
    """
    document_id = state["document_id"]
    question = state["patient_question"]
    triage_report = state.get("triage_report", "")
    urgency = state.get("urgency_level", "routine")
    iteration = state.get("iterations", 1)

    logger.info(
        f"Diagnosis Agent | doc_id={document_id} | "
        f"urgency={urgency} | iter={iteration}"
    )

    # Load clinical entities
    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}

    symptoms = entities.get("symptoms", [])
    diagnoses = entities.get("diagnoses", [])
    lab_values = entities.get("lab_values", [])
    vitals = entities.get("vitals", [])

    # Graph-based differential
    graph_differential = []
    graph_complications = []
    combined_path = get_combined_graph_path(document_id)

    if combined_path.exists():
        G = load_graph(combined_path)
        symptom_names = [
            s.get("name", "").lower() for s in symptoms
        ]
        graph_differential = get_differential_diagnosis(G, symptom_names)

        diag_names = [d.get("name", "").lower() for d in diagnoses]
        med_names = []
        clinical_picture = build_patient_clinical_picture(
            G, diag_names, med_names, symptom_names
        )
        graph_complications = clinical_picture.get(
            "complications_to_watch", []
        )[:5]

    # Build prompt strings
    symptoms_str = "\n".join([
        f"- {s.get('name')} "
        f"(severity: {s.get('severity','unknown')}, "
        f"duration: {s.get('duration','unknown')})"
        for s in symptoms
    ]) or "None documented"

    diagnoses_str = "\n".join([
        f"- {d.get('name')} [{d.get('status','unknown')}]"
        for d in diagnoses
    ]) or "None documented"

    labs_str = "\n".join([
        f"- {l.get('test_name')}: {l.get('value')} "
        f"{l.get('unit','')} [{l.get('interpretation','')}]"
        for l in lab_values
    ]) or "None documented"

    graph_diff_str = "\n".join([
        f"- {d.get('label', d.get('disease',''))}: "
        f"probability score {d.get('score',0):.2f}"
        for d in graph_differential[:5]
    ]) or "Graph not available"

    complications_str = "\n".join([
        f"- {c.get('from','')} → "
        f"{c.get('relation','')} → {c.get('to','')}"
        for c in graph_complications
    ]) or "None identified"

    # FIXED: moved newline join outside f-string
    vitals_str = "\n".join([
        f"- {v.get('name')}: {v.get('value')}"
        for v in vitals
    ]) or "None"

    system_prompt = """You are a senior physician performing clinical diagnosis.
Be systematic, evidence-based, and precise.
Always support conclusions with specific clinical findings."""

    user_prompt = f"""Clinical Question: {question}

Triage Context:
{triage_report}

SYMPTOMS:
{symptoms_str}

DOCUMENTED DIAGNOSES:
{diagnoses_str}

LAB VALUES:
{labs_str}

VITALS:
{vitals_str}

GRAPH-BASED DIFFERENTIAL DIAGNOSIS:
{graph_diff_str}

POTENTIAL COMPLICATIONS (from knowledge graph):
{complications_str}

Provide a comprehensive diagnostic assessment:

**PRIMARY ASSESSMENT:**
[Most likely diagnosis with confidence level and supporting evidence]

**DIFFERENTIAL DIAGNOSES:**
[Rank top 3-4 alternatives with supporting and opposing evidence]

**CRITICAL FINDINGS:**
[Any findings requiring immediate attention]

**COMPLICATIONS TO MONITOR:**
[Based on diagnoses — what complications must be watched]

**DIAGNOSTIC RECOMMENDATION:**
[What additional tests would confirm or rule out diagnoses]

**CLINICAL REASONING:**
[Step-by-step reasoning connecting symptoms to conclusions]"""

    diagnosis_report = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1200,
        temperature=0.1,
    )

    primary = diagnoses[0].get("name", "Undetermined") if diagnoses else "Undetermined"

    logger.info(
        f"Diagnosis Agent complete | "
        f"primary='{primary}' | "
        f"differential={len(graph_differential)}"
    )

    return {
        "diagnosis_report": diagnosis_report,
        "primary_diagnosis": primary,
        "differential_diagnoses": graph_differential[:5],
        "icd10_codes": entities_data.get("icd10_codes", {}),
        "agent_messages": [{
            "agent": "Diagnosis",
            "iteration": iteration,
            "primary_diagnosis": primary,
            "differential_count": len(graph_differential),
        }],
    }