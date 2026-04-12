from langgraph.graph import StateGraph, END
from backend.clinical_agents.clinical_state import ClinicalState
from backend.clinical_agents.triage_agent import triage_agent
from backend.clinical_agents.diagnosis_agent import diagnosis_agent
from backend.clinical_agents.pharmacist_agent import pharmacist_agent
from backend.clinical_agents.research_agent import research_agent
from backend.clinical_agents.safety_agent import safety_agent
from backend.logger import get_logger

logger = get_logger("clinical_agents.graph")


def route_after_triage(state: dict) -> str:
    """
    After triage — if emergency, skip to safety agent.
    Otherwise proceed to diagnosis.
    """
    if state.get("is_emergency", False):
        logger.warning("EMERGENCY routing — skipping to safety agent")
        return "safety"
    return "diagnosis"


def route_after_safety(state: dict) -> str:
    """
    After safety review — approve and end, or
    send back to specific agent for revision.
    """
    approved = state.get("safety_approved", False)
    revision = state.get("revision_needed", False)
    iterations = state.get("iterations", 1)
    max_iter = state.get("max_iterations", 3)

    if approved or not revision or iterations >= max_iter:
        logger.info(
            f"Safety routing → END | "
            f"approved={approved} | iter={iterations}"
        )
        return "end"

    target = state.get("revision_target", "diagnosis")
    logger.info(
        f"Safety routing → {target} revision | iter={iterations}"
    )

    revision_map = {
        "triage": "triage",
        "diagnosis": "diagnosis",
        "pharmacist": "pharmacist",
        "research": "research",
    }
    return revision_map.get(target, "diagnosis")


def build_clinical_graph():
    """
    Builds the 5-agent clinical LangGraph workflow.

    Normal flow:
    START → Triage → Diagnosis → Pharmacist → Research → Safety → END

    Emergency shortcut:
    START → Triage → Safety → END

    Revision loop (max 2):
    Safety → [specific agent] → Safety → END
    """
    logger.info("Building 5-agent clinical LangGraph")

    workflow = StateGraph(ClinicalState)

    # Add all 5 agent nodes
    workflow.add_node("triage", triage_agent)
    workflow.add_node("diagnosis", diagnosis_agent)
    workflow.add_node("pharmacist", pharmacist_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("safety", safety_agent)

    # Entry point
    workflow.set_entry_point("triage")

    # Triage → conditional routing
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "diagnosis": "diagnosis",
            "safety": "safety",
        }
    )

    # Normal sequential flow
    workflow.add_edge("diagnosis", "pharmacist")
    workflow.add_edge("pharmacist", "research")
    workflow.add_edge("research", "safety")

    # Safety → conditional routing
    workflow.add_conditional_edges(
        "safety",
        route_after_safety,
        {
            "end": END,
            "triage": "triage",
            "diagnosis": "diagnosis",
            "pharmacist": "pharmacist",
            "research": "research",
        }
    )

    app = workflow.compile()
    logger.info("Clinical graph compiled successfully")
    return app


# Build once at module load
clinical_graph = build_clinical_graph()