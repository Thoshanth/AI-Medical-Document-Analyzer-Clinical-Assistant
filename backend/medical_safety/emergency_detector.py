import re
from backend.logger import get_logger

logger = get_logger("medical_safety.emergency")

# Emergency symptom combinations — any of these clusters
# indicate a potential life-threatening emergency
EMERGENCY_CLUSTERS = [
    # Cardiac emergency
    {
        "name": "possible_cardiac_emergency",
        "keywords": ["chest pain", "chest tightness", "heart attack"],
        "supporting": ["sweating", "arm pain", "jaw pain", "nausea"],
        "threshold": 1,
    },
    # Stroke
    {
        "name": "possible_stroke",
        "keywords": ["face drooping", "arm weakness", "speech difficulty",
                     "sudden numbness", "sudden confusion", "stroke"],
        "supporting": ["sudden", "one side", "severe headache"],
        "threshold": 1,
    },
    # Respiratory emergency
    {
        "name": "respiratory_emergency",
        "keywords": ["can't breathe", "cannot breathe",
                     "difficulty breathing", "respiratory arrest",
                     "choking", "anaphylaxis"],
        "supporting": ["severe", "sudden", "lips turning blue", "cyanosis"],
        "threshold": 1,
    },
    # Severe bleeding
    {
        "name": "severe_bleeding",
        "keywords": ["severe bleeding", "bleeding won't stop",
                     "hemorrhage", "blood loss"],
        "supporting": ["large amount", "gushing", "uncontrolled"],
        "threshold": 1,
    },
    # Unconsciousness
    {
        "name": "loss_of_consciousness",
        "keywords": ["unconscious", "unresponsive", "passed out",
                     "fainted", "not breathing", "cardiac arrest"],
        "supporting": [],
        "threshold": 1,
    },
    # Overdose
    {
        "name": "overdose",
        "keywords": ["overdose", "took too many pills",
                     "swallowed too much", "poisoning"],
        "supporting": ["medication", "drug", "pills", "tablets"],
        "threshold": 1,
    },
    # Suicide
    {
        "name": "suicide_or_self_harm",
        "keywords": ["want to die", "kill myself", "end my life",
                     "suicide", "self harm", "hurt myself"],
        "supporting": [],
        "threshold": 1,
    },
]

EMERGENCY_RESPONSE_TEMPLATE = """🚨 **MEDICAL EMERGENCY DETECTED**

Based on the symptoms described, this may be a medical emergency requiring **immediate action**.

**⚡ CALL EMERGENCY SERVICES NOW:**
- **India:** 112 or 108 (Ambulance)
- **US/Canada:** 911
- **UK:** 999
- **International:** 112

**Do NOT wait for AI assistance in a medical emergency.**

{specific_guidance}

---
*This AI system detected emergency indicators in your message. 
Always call emergency services for life-threatening situations.*
"""

SPECIFIC_GUIDANCE = {
    "possible_cardiac_emergency": """
**If chest pain/heart attack symptoms:**
- Call ambulance immediately
- Have patient sit or lie down — do not let them walk
- Loosen tight clothing
- If available and not allergic: give aspirin 300mg to chew
- Begin CPR if patient loses consciousness and stops breathing
""",
    "possible_stroke": """
**Remember FAST for stroke:**
- **F**ace — is it drooping?
- **A**rm — is one arm weak?
- **S**peech — is it slurred?
- **T**ime — call emergency services NOW
- Note the time symptoms started — crucial for treatment
""",
    "respiratory_emergency": """
**For breathing difficulty:**
- Call ambulance immediately
- Keep patient upright if possible
- If allergic reaction: use EpiPen if available
- Loosen any tight clothing around neck/chest
- Stay with patient until help arrives
""",
    "loss_of_consciousness": """
**For unconscious person:**
- Check if they are breathing
- If not breathing: begin CPR immediately
- Call emergency services
- Place in recovery position if breathing
- Do not leave them alone
""",
    "overdose": """
**For suspected overdose:**
- Call emergency services immediately
- Note what was taken and how much if known
- Keep patient awake if possible
- Do not induce vomiting unless instructed by medical professional
- Save medication containers/labels for paramedics
""",
    "suicide_or_self_harm": """
**If someone is in crisis:**
- **Vandrevala Foundation Helpline (India):** 1860-2662-345 (24/7)
- **iCall:** 9152987821
- **AASRA:** 9820466627
- **International:** findahelpline.com

Stay with the person. Listen without judgment.
Remove access to harmful items if safe to do so.
Call emergency services if immediate danger.
""",
}


def detect_emergency(text: str) -> dict:
    """
    Scans input text for emergency indicators.

    Returns:
    {
        "is_emergency": bool,
        "emergency_type": str or None,
        "emergency_response": str or None,
        "confidence": "high" | "medium" | "low"
    }
    """
    text_lower = text.lower()
    detected = []

    for cluster in EMERGENCY_CLUSTERS:
        keyword_matches = sum(
            1 for kw in cluster["keywords"]
            if kw in text_lower
        )

        if keyword_matches >= cluster["threshold"]:
            supporting_matches = sum(
                1 for sw in cluster.get("supporting", [])
                if sw in text_lower
            )
            confidence = (
                "high" if supporting_matches >= 2
                else "medium" if supporting_matches >= 1
                else "low"
            )
            detected.append({
                "type": cluster["name"],
                "confidence": confidence,
                "keyword_matches": keyword_matches,
            })

    if not detected:
        return {
            "is_emergency": False,
            "emergency_type": None,
            "emergency_response": None,
            "confidence": None,
        }

    # Use highest confidence detection
    best = max(detected, key=lambda x: (
        {"high": 3, "medium": 2, "low": 1}[x["confidence"]]
    ))

    emergency_type = best["type"]
    specific = SPECIFIC_GUIDANCE.get(emergency_type, "")
    response = EMERGENCY_RESPONSE_TEMPLATE.format(
        specific_guidance=specific
    )

    logger.warning(
        f"EMERGENCY DETECTED | "
        f"type={emergency_type} | "
        f"confidence={best['confidence']}"
    )

    return {
        "is_emergency": True,
        "emergency_type": emergency_type,
        "emergency_response": response,
        "confidence": best["confidence"],
    }