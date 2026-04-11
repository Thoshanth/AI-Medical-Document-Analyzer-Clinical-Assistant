import json
from backend.logger import get_logger

logger = get_logger("ingestion.fhir")


def parse_fhir_json(file_path: str) -> str:
    """
    Parses a FHIR JSON file and converts it to
    human-readable medical text that the LLM can understand.

    FHIR resources we handle:
    - Patient: demographics
    - Condition: diagnoses
    - Medication/MedicationRequest: prescriptions
    - Observation: lab results, vitals
    - DiagnosticReport: imaging, pathology
    - AllergyIntolerance: allergies
    - Encounter: visit records
    """
    logger.info(f"Parsing FHIR JSON | file='{file_path}'")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    resource_type = data.get("resourceType", "Unknown")
    logger.info(f"FHIR resource type: {resource_type}")

    # Route to specific parser based on resource type
    parsers = {
        "Patient": _parse_patient,
        "Bundle": _parse_bundle,
        "Condition": _parse_condition,
        "MedicationRequest": _parse_medication_request,
        "Observation": _parse_observation,
        "DiagnosticReport": _parse_diagnostic_report,
        "AllergyIntolerance": _parse_allergy,
        "Encounter": _parse_encounter,
    }

    parser = parsers.get(resource_type, _parse_generic)
    text = parser(data)

    logger.info(f"FHIR parsing complete | chars={len(text)}")
    return text


def _parse_patient(data: dict) -> str:
    lines = ["=== PATIENT RECORD ==="]

    # Name
    names = data.get("name", [])
    if names:
        name = names[0]
        full_name = " ".join(
            name.get("given", []) + [name.get("family", "")]
        ).strip()
        lines.append(f"Patient Name: {full_name}")

    # Demographics
    if dob := data.get("birthDate"):
        lines.append(f"Date of Birth: {dob}")
    if gender := data.get("gender"):
        lines.append(f"Gender: {gender.title()}")

    # Contact
    telecoms = data.get("telecom", [])
    for t in telecoms:
        lines.append(f"Contact ({t.get('use', 'general')}): {t.get('value', '')}")

    # Address
    addresses = data.get("address", [])
    for addr in addresses:
        city = addr.get("city", "")
        state = addr.get("state", "")
        country = addr.get("country", "")
        lines.append(f"Address: {city}, {state}, {country}".strip(", "))

    # Identifiers (MRN etc.)
    identifiers = data.get("identifier", [])
    for ident in identifiers:
        id_type = ident.get("type", {}).get("text", "ID")
        id_value = ident.get("value", "")
        lines.append(f"{id_type}: {id_value}")

    return "\n".join(lines)


def _parse_condition(data: dict) -> str:
    lines = ["=== MEDICAL CONDITION ==="]

    code = data.get("code", {})
    condition_text = code.get("text", "")
    codings = code.get("coding", [])

    if condition_text:
        lines.append(f"Condition: {condition_text}")

    for coding in codings:
        lines.append(
            f"Code: {coding.get('system', '')} | "
            f"{coding.get('code', '')} | "
            f"{coding.get('display', '')}"
        )

    if severity := data.get("severity", {}).get("text"):
        lines.append(f"Severity: {severity}")

    if status := data.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"):
        lines.append(f"Clinical Status: {status}")

    if onset := data.get("onsetDateTime"):
        lines.append(f"Onset Date: {onset}")

    if note := data.get("note", []):
        lines.append(f"Notes: {note[0].get('text', '')}")

    return "\n".join(lines)


def _parse_medication_request(data: dict) -> str:
    lines = ["=== MEDICATION PRESCRIPTION ==="]

    med = data.get("medicationCodeableConcept", {})
    med_name = med.get("text", "") or med.get("coding", [{}])[0].get("display", "")
    if med_name:
        lines.append(f"Medication: {med_name}")

    if status := data.get("status"):
        lines.append(f"Status: {status}")

    if intent := data.get("intent"):
        lines.append(f"Intent: {intent}")

    # Dosage instructions
    dosages = data.get("dosageInstruction", [])
    for i, dosage in enumerate(dosages, 1):
        text = dosage.get("text", "")
        if text:
            lines.append(f"Dosage {i}: {text}")

        timing = dosage.get("timing", {}).get("code", {}).get("text", "")
        if timing:
            lines.append(f"Timing: {timing}")

        route = dosage.get("route", {}).get("text", "")
        if route:
            lines.append(f"Route: {route}")

    return "\n".join(lines)


def _parse_observation(data: dict) -> str:
    lines = ["=== CLINICAL OBSERVATION / LAB RESULT ==="]

    code = data.get("code", {})
    obs_name = code.get("text", "") or code.get("coding", [{}])[0].get("display", "")
    if obs_name:
        lines.append(f"Observation: {obs_name}")

    if status := data.get("status"):
        lines.append(f"Status: {status}")

    # Numeric value
    if value := data.get("valueQuantity"):
        val = value.get("value", "")
        unit = value.get("unit", "")
        lines.append(f"Value: {val} {unit}".strip())

    # String value
    if value_str := data.get("valueString"):
        lines.append(f"Result: {value_str}")

    # Reference range
    ranges = data.get("referenceRange", [])
    for r in ranges:
        low = r.get("low", {}).get("value", "")
        high = r.get("high", {}).get("value", "")
        unit = r.get("low", {}).get("unit", "")
        if low or high:
            lines.append(f"Reference Range: {low} - {high} {unit}".strip())

    # Interpretation (normal/abnormal/critical)
    interpretations = data.get("interpretation", [])
    for interp in interpretations:
        text = interp.get("text", "") or interp.get(
            "coding", [{}]
        )[0].get("display", "")
        if text:
            lines.append(f"Interpretation: {text}")

    if effective := data.get("effectiveDateTime"):
        lines.append(f"Date: {effective}")

    return "\n".join(lines)


def _parse_diagnostic_report(data: dict) -> str:
    lines = ["=== DIAGNOSTIC REPORT ==="]

    code = data.get("code", {})
    report_name = code.get("text", "")
    if report_name:
        lines.append(f"Report Type: {report_name}")

    if status := data.get("status"):
        lines.append(f"Status: {status}")

    if issued := data.get("issued"):
        lines.append(f"Issued: {issued}")

    # Conclusion
    if conclusion := data.get("conclusion"):
        lines.append(f"Conclusion: {conclusion}")

    # Results referenced
    results = data.get("result", [])
    if results:
        lines.append(f"Number of Results: {len(results)}")

    return "\n".join(lines)


def _parse_allergy(data: dict) -> str:
    lines = ["=== ALLERGY / INTOLERANCE ==="]

    substance = data.get("code", {}).get("text", "")
    if substance:
        lines.append(f"Substance: {substance}")

    if allergy_type := data.get("type"):
        lines.append(f"Type: {allergy_type}")

    if category := data.get("category", []):
        lines.append(f"Category: {', '.join(category)}")

    if criticality := data.get("criticality"):
        lines.append(f"Criticality: {criticality}")

    reactions = data.get("reaction", [])
    for r in reactions:
        manifestations = r.get("manifestation", [])
        for m in manifestations:
            lines.append(f"Reaction: {m.get('text', '')}")
        if severity := r.get("severity"):
            lines.append(f"Severity: {severity}")

    return "\n".join(lines)


def _parse_encounter(data: dict) -> str:
    lines = ["=== CLINICAL ENCOUNTER ==="]

    if status := data.get("status"):
        lines.append(f"Status: {status}")

    enc_class = data.get("class", {}).get("display", "")
    if enc_class:
        lines.append(f"Encounter Type: {enc_class}")

    types = data.get("type", [])
    for t in types:
        type_text = t.get("text", "")
        if type_text:
            lines.append(f"Visit Type: {type_text}")

    period = data.get("period", {})
    if start := period.get("start"):
        lines.append(f"Start: {start}")
    if end := period.get("end"):
        lines.append(f"End: {end}")

    reasons = data.get("reasonCode", [])
    for r in reasons:
        reason_text = r.get("text", "")
        if reason_text:
            lines.append(f"Reason: {reason_text}")

    return "\n".join(lines)


def _parse_bundle(data: dict) -> str:
    """
    A Bundle contains multiple FHIR resources.
    Parse each entry and combine.
    """
    lines = ["=== FHIR BUNDLE (Multiple Records) ==="]
    entries = data.get("entry", [])
    logger.info(f"FHIR Bundle | entries={len(entries)}")

    for i, entry in enumerate(entries):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "Unknown")

        parsers = {
            "Patient": _parse_patient,
            "Condition": _parse_condition,
            "MedicationRequest": _parse_medication_request,
            "Observation": _parse_observation,
            "DiagnosticReport": _parse_diagnostic_report,
            "AllergyIntolerance": _parse_allergy,
            "Encounter": _parse_encounter,
        }

        parser = parsers.get(resource_type, _parse_generic)
        entry_text = parser(resource)
        lines.append(f"\n--- Record {i+1}: {resource_type} ---")
        lines.append(entry_text)

    return "\n".join(lines)


def _parse_generic(data: dict) -> str:
    """Fallback for unknown FHIR resource types."""
    lines = [f"=== FHIR RECORD: {data.get('resourceType', 'Unknown')} ==="]
    lines.append(f"Resource ID: {data.get('id', 'N/A')}")
    lines.append(f"Status: {data.get('status', 'N/A')}")
    return "\n".join(lines)