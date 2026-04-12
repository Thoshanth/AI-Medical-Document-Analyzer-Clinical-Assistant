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
    medications = entities.get("medications", [])

    # Graph-based differential
    graph_differential = []
    graph_complications = []
    combined_path = get_combined_graph_path(document_id)

    if combined_path.exists():
        try:
            G = load_graph(combined_path)
            symptom_names = [
                s.get("name", "").lower()
                for s in symptoms
                if s.get("name")
            ]
            graph_differential = get_differential_diagnosis(
                G, symptom_names
            )

            diag_names = [
                d.get("name", "").lower()
                for d in diagnoses
                if d.get("name")
            ]
            med_names = [
                m.get("name", "").lower()
                for m in medications
                if m.get("name")
            ]

            clinical_picture = build_patient_clinical_picture(
                G, diag_names, med_names, symptom_names
            )
            graph_complications = clinical_picture.get(
                "complications_to_watch", []
            )[:5]

        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")

    # ── Build prompt strings safely (no backslash in f-strings) ──

    if symptoms:
        symptoms_str = "\n".join([
            f"- {s.get('name', 'Unknown')} "
            f"(severity: {s.get('severity', 'unknown')}, "
            f"duration: {s.get('duration', 'unknown')})"
            for s in symptoms
        ])
    else:
        symptoms_str = "None documented"

    if diagnoses:
        diagnoses_str = "\n".join([
            f"- {d.get('name', 'Unknown')} "
            f"[{d.get('status', 'unknown')}]"
            for d in diagnoses
        ])
    else:
        diagnoses_str = "None documented"

    if lab_values:
        labs_str = "\n".join([
            f"- {l.get('test_name', 'Unknown')}: "
            f"{l.get('value', 'N/A')} "
            f"{l.get('unit', '')} "
            f"[{l.get('interpretation', 'unknown')}]"
            for l in lab_values
        ])
    else:
        labs_str = "None documented"

    if vitals:
        vitals_str = "\n".join([
            f"- {v.get('name', 'Unknown')}: "
            f"{v.get('value', 'N/A')} "
            f"[{v.get('status', 'unknown')}]"
            for v in vitals
        ])
    else:
        vitals_str = "None documented"

    if graph_differential:
        graph_diff_str = "\n".join([
            f"- {d.get('label', d.get('disease', 'Unknown'))}: "
            f"probability score {d.get('score', 0):.2f}"
            for d in graph_differential[:5]
        ])
    else:
        graph_diff_str = "Graph not available"

    if graph_complications:
        complications_str = "\n".join([
            f"- {c.get('from', '')} "
            f"→ {c.get('relation', '')} "
            f"→ {c.get('to', '')}"
            for c in graph_complications
        ])
    else:
        complications_str = "None identified"

    # ── Build and send prompt ─────────────────────────────────────

    system_prompt = """You are a senior physician performing clinical diagnosis.
Be systematic, evidence-based, and precise.
Always support conclusions with specific clinical findings."""

    user_prompt = (
        f"Clinical Question: {question}\n\n"
        f"Triage Context:\n{triage_report}\n\n"
        f"SYMPTOMS:\n{symptoms_str}\n\n"
        f"DOCUMENTED DIAGNOSES:\n{diagnoses_str}\n\n"
        f"LAB VALUES:\n{labs_str}\n\n"
        f"VITAL SIGNS:\n{vitals_str}\n\n"
        f"GRAPH-BASED DIFFERENTIAL DIAGNOSIS:\n{graph_diff_str}\n\n"
        f"POTENTIAL COMPLICATIONS (from knowledge graph):\n{complications_str}\n\n"
        f"Provide a comprehensive diagnostic assessment:\n\n"
        f"**PRIMARY ASSESSMENT:**\n"
        f"[Most likely diagnosis with confidence level and supporting evidence]\n\n"
        f"**DIFFERENTIAL DIAGNOSES:**\n"
        f"[Rank top 3-4 alternatives with supporting and opposing evidence]\n\n"
        f"**CRITICAL FINDINGS:**\n"
        f"[Any findings requiring immediate attention]\n\n"
        f"**COMPLICATIONS TO MONITOR:**\n"
        f"[Based on diagnoses — what complications must be watched]\n\n"
        f"**DIAGNOSTIC RECOMMENDATION:**\n"
        f"[What additional tests would confirm or rule out diagnoses]\n\n"
        f"**CLINICAL REASONING:**\n"
        f"[Step-by-step reasoning connecting symptoms to conclusions]"
    )

    try:
        diagnosis_report = chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.1,
        )
    except Exception as e:
        logger.error(f"LLM call failed in diagnosis agent: {e}")
        diagnosis_report = (
            "Diagnosis generation failed. "
            "Please retry or review clinical entities manually."
        )

    primary = (
        diagnoses[0].get("name", "Undetermined")
        if diagnoses
        else "Undetermined"
    )

    logger.info(
        f"Diagnosis Agent complete | "
        f"primary='{primary}' | "
        f"differential={len(graph_differential)}"
    )

    return {
        "diagnosis_report": diagnosis_report,
        "primary_diagnosis": primary,
        "differential_diagnoses": graph_differential[:5],
        "icd10_codes": entities_data.get("icd10_codes", {}) if entities_data else {},
        "agent_messages": [{
            "agent": "Diagnosis",
            "iteration": iteration,
            "primary_diagnosis": primary,
            "differential_count": len(graph_differential),
        }],
    }