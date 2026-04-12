from backend.llm_client import chat_completion
from backend.medical_rag.rag_pipeline import medical_rag_query
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.logger import get_logger

logger = get_logger("clinical_agents.research")


def research_agent(state: dict) -> dict:
    """
    Agent 4 — Research Agent

    Medical evidence specialist. Searches indexed documents
    and medical knowledge to find evidence supporting the
    diagnosis and treatment plan.

    Reads:  document_id, primary_diagnosis, diagnosis_report
    Writes: research_findings, evidence_level, clinical_guidelines
    """
    document_id = state["document_id"]
    question = state["patient_question"]
    primary_diagnosis = state.get("primary_diagnosis", "")
    diagnosis_report = state.get("diagnosis_report", "")
    iteration = state.get("iterations", 1)

    logger.info(
        f"Research Agent | doc_id={document_id} | "
        f"diagnosis='{primary_diagnosis}' | iter={iteration}"
    )

    # Use Stage 4 Medical RAG to search for evidence
    research_query = (
        f"Evidence-based treatment and management for "
        f"{primary_diagnosis}. "
        f"Clinical guidelines and recommendations."
    )

    rag_result = {}
    rag_answer = ""
    try:
        rag_result = medical_rag_query(
            question=research_query,
            document_id=document_id,
            top_k=5,
            include_kb=True,
        )
        rag_answer = rag_result.get("answer", "")
        logger.info(
            f"RAG research complete | chars={len(rag_answer)}"
        )
    except Exception as e:
        logger.warning(f"RAG research failed: {e}")
        rag_answer = "Unable to retrieve evidence from documents."

    # Load clinical entities for context
    entities_data = get_clinical_entities(document_id)
    entities = entities_data.get("entities", {}) if entities_data else {}
    diagnoses = entities.get("diagnoses", [])

    system_prompt = """You are a medical research specialist and 
evidence-based medicine expert. Synthesize clinical evidence 
and guidelines to support diagnostic and treatment decisions."""

    user_prompt = f"""Clinical Question: {question}
Primary Diagnosis: {primary_diagnosis}

Diagnosis Context:
{diagnosis_report[:500] if diagnosis_report else 'Not available'}

Evidence from Document Search:
{rag_answer or 'No evidence retrieved from documents'}

Known Diagnoses: {', '.join([d.get('name','') for d in diagnoses])}

Provide an evidence-based research summary:

**EVIDENCE SUMMARY:**
[Key evidence supporting or challenging the diagnosis]

**CLINICAL GUIDELINES:**
[Relevant clinical practice guidelines for this condition]

**EVIDENCE LEVEL:**
[Grade the evidence: A (strong RCTs) | B (good studies) | 
C (limited evidence) | Expert Opinion]

**TREATMENT EVIDENCE:**
[Evidence for recommended treatments]

**GAPS IN EVIDENCE:**
[What is unknown or uncertain]

**RESEARCH RECOMMENDATIONS:**
[What additional evidence or tests would help]

Base all statements on provided document evidence or established guidelines."""

    research_findings = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
        temperature=0.1,
    )

    # Extract evidence level from response
    evidence_level = "C"
    if "evidence level: a" in research_findings.lower():
        evidence_level = "A"
    elif "evidence level: b" in research_findings.lower():
        evidence_level = "B"
    elif "expert opinion" in research_findings.lower():
        evidence_level = "Expert Opinion"

    logger.info(
        f"Research Agent complete | "
        f"evidence_level={evidence_level} | "
        f"chars={len(research_findings)}"
    )

    return {
        "research_findings": research_findings,
        "evidence_level": evidence_level,
        "clinical_guidelines": [
            f"Evidence from document search for {primary_diagnosis}"
        ],
        "agent_messages": [{
            "agent": "Research",
            "iteration": iteration,
            "evidence_level": evidence_level,
            "rag_used": bool(rag_answer),
        }],
    }