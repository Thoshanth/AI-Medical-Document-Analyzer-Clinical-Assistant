# Medical node types
MEDICAL_NODE_TYPES = {
    "SYMPTOM": "symptom",
    "DISEASE": "disease",
    "DRUG": "drug",
    "LAB_TEST": "lab_test",
    "PROCEDURE": "procedure",
    "ANATOMY": "anatomy",
    "FINDING": "finding",
    "CONDITION": "condition",
    "RISK_FACTOR": "risk_factor",
}

# Medical relationship types
MEDICAL_EDGE_TYPES = {
    "INDICATES": "indicates",
    "CAUSED_BY": "caused_by",
    "TREATED_BY": "treated_by",
    "DIAGNOSED_BY": "diagnosed_by",
    "MONITORED_BY": "monitored_by",
    "CONTRAINDICATED_IN": "contraindicated_in",
    "INTERACTS_WITH": "interacts_with",
    "AFFECTS": "affects",
    "COMPLICATION_OF": "complication_of",
    "RISK_FACTOR_FOR": "risk_factor_for",
    "ASSOCIATED_WITH": "associated_with",
    "REQUIRES": "requires",
    "PREVENTS": "prevents",
    "WORSENS": "worsens",
}

# Severity levels for relationships
SEVERITY_LEVELS = {
    "critical": 4,
    "major": 3,
    "moderate": 2,
    "minor": 1,
    "unknown": 0,
}