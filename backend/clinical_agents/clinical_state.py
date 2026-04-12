from typing import TypedDict, Annotated
import operator


class ClinicalState(TypedDict):
    """
    Shared state passed between all 5 clinical agents.
    Each agent reads the full state and writes to their section.
    Annotated with operator.add means list fields are appended.
    """

    # Input
    document_id: int
    patient_question: str

    # Agent 1 — Triage
    triage_report: str
    urgency_level: str          # emergency | urgent | routine
    triage_alerts: list[str]
    is_emergency: bool

    # Agent 2 — Diagnosis
    diagnosis_report: str
    primary_diagnosis: str
    differential_diagnoses: list[dict]
    icd10_codes: dict

    # Agent 3 — Pharmacist
    pharmacist_report: str
    drug_alerts: list[dict]
    interaction_summary: str
    medication_safe: bool

    # Agent 4 — Research
    research_findings: str
    evidence_level: str         # A | B | C | expert_opinion
    clinical_guidelines: list[str]

    # Agent 5 — Safety
    safety_report: str
    safety_approved: bool
    safety_concerns: list[str]
    revision_needed: bool
    revision_target: str        # which agent needs to revise

    # Control
    iterations: int
    max_iterations: int
    final_answer: str

    # Full trace
    agent_messages: Annotated[list[dict], operator.add]