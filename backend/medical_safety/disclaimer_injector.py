from backend.logger import get_logger

logger = get_logger("medical_safety.disclaimer")

# Different disclaimer levels based on urgency and content
DISCLAIMERS = {
    "emergency": """
🚨 **EMERGENCY MEDICAL DISCLAIMER**
This is AI-generated information. **Call emergency services immediately (112/911)** 
for life-threatening conditions. Do not rely on AI in emergencies.
""",

    "clinical": """
⚕️ **CLINICAL DISCLAIMER**
This AI-generated clinical assessment is for **decision support only**.
It does not constitute medical advice, diagnosis, or treatment.
All clinical decisions must be made by **licensed healthcare professionals**.
Verify all information against authoritative medical sources.
""",

    "medication": """
💊 **MEDICATION SAFETY DISCLAIMER**
Medication information is AI-generated and **must be verified** by a 
licensed pharmacist or physician before clinical use.
Drug interactions and contraindications require professional evaluation.
Never adjust medications based solely on AI recommendations.
""",

    "lab": """
🔬 **LABORATORY DISCLAIMER**
Lab result interpretations are AI-generated for educational purposes.
Clinical correlation with patient history and examination is required.
Critical values require **immediate physician notification**.
All interpretations must be validated by a qualified pathologist or clinician.
""",

    "general": """
ℹ️ **MEDICAL INFORMATION DISCLAIMER**
This information is provided by an AI system for educational purposes only.
It is **not a substitute for professional medical advice**.
Always consult a qualified healthcare provider for medical decisions.
""",

    "research": """
📚 **RESEARCH DISCLAIMER**
Evidence summaries are AI-generated and may not reflect the most current 
clinical guidelines. Always consult up-to-date medical literature and 
clinical practice guidelines from authoritative sources (WHO, CDC, NICE, etc.).
""",
}


def inject_disclaimer(
    text: str,
    disclaimer_type: str = "general",
    urgency_level: str = "routine",
) -> str:
    """
    Injects appropriate medical disclaimer into response text.

    Disclaimer type selection:
    - emergency urgency → emergency disclaimer
    - contains medication info → medication disclaimer
    - contains lab values → lab disclaimer
    - clinical report → clinical disclaimer
    - default → general disclaimer
    """
    # Override disclaimer type based on urgency
    if urgency_level == "emergency":
        disclaimer_type = "emergency"

    disclaimer = DISCLAIMERS.get(
        disclaimer_type,
        DISCLAIMERS["general"]
    )

    # Add disclaimer at the end
    result = f"{text}\n\n---\n{disclaimer.strip()}"

    logger.debug(
        f"Disclaimer injected | type={disclaimer_type}"
    )

    return result


def select_disclaimer_type(
    text: str,
    urgency_level: str = "routine",
) -> str:
    """
    Automatically selects the most appropriate disclaimer
    type based on response content.
    """
    if urgency_level == "emergency":
        return "emergency"

    text_lower = text.lower()

    if any(word in text_lower for word in [
        "medication", "drug", "dosage", "prescription",
        "tablet", "capsule", "mg", "interaction"
    ]):
        return "medication"

    if any(word in text_lower for word in [
        "lab", "blood test", "result", "value",
        "normal range", "critical", "wbc", "hemoglobin"
    ]):
        return "lab"

    if any(word in text_lower for word in [
        "diagnosis", "assessment", "plan", "soap",
        "treatment", "clinical", "patient"
    ]):
        return "clinical"

    if any(word in text_lower for word in [
        "study", "research", "evidence", "guideline",
        "trial", "published"
    ]):
        return "research"

    return "general"