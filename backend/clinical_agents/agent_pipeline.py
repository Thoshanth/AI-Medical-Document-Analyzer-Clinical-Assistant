from backend.clinical_agents.clinical_graph import clinical_graph
from backend.logger import get_logger
from datetime import datetime

logger = get_logger("clinical_agents.pipeline")


def run_clinical_agents(
    document_id: int,
    question: str,
    max_iterations: int = 3,
    show_agent_trace: bool = False,
) -> dict:
    """
    Runs the complete 5-agent clinical analysis system.

    Initializes clinical state and invokes the LangGraph
    workflow. All 5 agents run in sequence with safety
    review and optional revision loops.

    Returns comprehensive clinical assessment with
    mandatory medical disclaimer.
    """
    logger.info(
        f"Clinical agents starting | "
        f"doc_id={document_id} | "
        f"question='{question[:60]}'"
    )

    initial_state = {
        "document_id": document_id,
        "patient_question": question,
        "triage_report": "",
        "urgency_level": "routine",
        "triage_alerts": [],
        "is_emergency": False,
        "diagnosis_report": "",
        "primary_diagnosis": "",
        "differential_diagnoses": [],
        "icd10_codes": {},
        "pharmacist_report": "",
        "drug_alerts": [],
        "interaction_summary": "",
        "medication_safe": True,
        "research_findings": "",
        "evidence_level": "C",
        "clinical_guidelines": [],
        "safety_report": "",
        "safety_approved": False,
        "safety_concerns": [],
        "revision_needed": False,
        "revision_target": "",
        "iterations": 1,
        "max_iterations": max_iterations,
        "final_answer": "",
        "agent_messages": [],
    }

    logger.info("Invoking clinical LangGraph workflow")
    final_state = clinical_graph.invoke(initial_state)

    # Build response
    response = {
        "document_id": document_id,
        "question": question,
        "generated_at": datetime.utcnow().isoformat(),
        "urgency_level": final_state.get("urgency_level", "routine"),
        "is_emergency": final_state.get("is_emergency", False),
        "primary_diagnosis": final_state.get("primary_diagnosis", ""),
        "medication_safe": final_state.get("medication_safe", True),
        "safety_approved": final_state.get("safety_approved", False),
        "evidence_level": final_state.get("evidence_level", "C"),
        "iterations_used": final_state.get("iterations", 1) - 1,
        "final_answer": final_state.get("final_answer", ""),
        "agents_used": [
            "Triage", "Diagnosis", "Pharmacist", "Research", "Safety"
        ],
        "critical_alerts": (
            final_state.get("triage_alerts", []) +
            [
                f"{a.get('drugs')}: {a.get('effect','')[:50]}"
                for a in final_state.get("drug_alerts", [])
                if a.get("severity") == "major"
            ]
        ),
    }

    if show_agent_trace:
        response["agent_trace"] = final_state.get("agent_messages", [])
        response["triage_report"] = final_state.get("triage_report", "")
        response["diagnosis_report"] = final_state.get("diagnosis_report", "")
        response["pharmacist_report"] = final_state.get("pharmacist_report", "")
        response["research_findings"] = final_state.get("research_findings", "")
        response["safety_report"] = final_state.get("safety_report", "")

    logger.info(
        f"Clinical agents complete | "
        f"urgency={response['urgency_level']} | "
        f"approved={response['safety_approved']} | "
        f"iterations={response['iterations_used']}"
    )

    return response