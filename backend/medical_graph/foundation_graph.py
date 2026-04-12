import networkx as nx
from backend.medical_graph.graph_schema import MEDICAL_NODE_TYPES
from backend.logger import get_logger

logger = get_logger("medical_graph.foundation")


def build_foundation_graph() -> nx.DiGraph:
    """
    Builds a pre-built medical knowledge graph from
    curated medical knowledge.

    This foundation graph exists before any document
    is uploaded. It contains core medical relationships
    that provide clinical context for document analysis.

    Covers: Diabetes, Hypertension, Pneumonia, Anemia,
    Hypothyroidism and their relationships to drugs,
    labs, symptoms, and complications.
    """
    logger.info("Building medical foundation knowledge graph")
    G = nx.DiGraph()

    # ── Add all nodes ──────────────────────────────────────────
    nodes = [
        # Diseases
        ("type_2_diabetes", {"type": "disease", "label": "Type 2 Diabetes", "icd10": "E11.9"}),
        ("hypertension", {"type": "disease", "label": "Hypertension", "icd10": "I10"}),
        ("pneumonia", {"type": "disease", "label": "Pneumonia", "icd10": "J18.9"}),
        ("anemia", {"type": "disease", "label": "Anemia", "icd10": "D64.9"}),
        ("hypothyroidism", {"type": "disease", "label": "Hypothyroidism", "icd10": "E03.9"}),
        ("ckd", {"type": "disease", "label": "Chronic Kidney Disease", "icd10": "N18.9"}),
        ("heart_failure", {"type": "disease", "label": "Heart Failure", "icd10": "I50.9"}),
        ("myocardial_infarction", {"type": "disease", "label": "Myocardial Infarction", "icd10": "I21.9"}),
        ("diabetic_nephropathy", {"type": "disease", "label": "Diabetic Nephropathy", "icd10": "N08"}),
        ("diabetic_retinopathy", {"type": "disease", "label": "Diabetic Retinopathy", "icd10": "E11.319"}),
        ("diabetic_neuropathy", {"type": "disease", "label": "Diabetic Neuropathy", "icd10": "E11.40"}),
        ("lactic_acidosis", {"type": "condition", "label": "Lactic Acidosis", "icd10": "E87.2"}),

        # Symptoms
        ("chest_pain", {"type": "symptom", "label": "Chest Pain"}),
        ("dyspnea", {"type": "symptom", "label": "Shortness of Breath / Dyspnea"}),
        ("fever", {"type": "symptom", "label": "Fever"}),
        ("fatigue", {"type": "symptom", "label": "Fatigue"}),
        ("cough", {"type": "symptom", "label": "Cough"}),
        ("polyuria", {"type": "symptom", "label": "Frequent Urination / Polyuria"}),
        ("polydipsia", {"type": "symptom", "label": "Excessive Thirst / Polydipsia"}),
        ("headache", {"type": "symptom", "label": "Headache"}),
        ("dizziness", {"type": "symptom", "label": "Dizziness"}),
        ("edema", {"type": "symptom", "label": "Edema / Swelling"}),
        ("pallor", {"type": "symptom", "label": "Pallor / Pale Skin"}),
        ("weight_gain", {"type": "symptom", "label": "Weight Gain"}),
        ("cold_intolerance", {"type": "symptom", "label": "Cold Intolerance"}),

        # Drugs
        ("metformin", {"type": "drug", "label": "Metformin", "class": "Biguanide"}),
        ("insulin", {"type": "drug", "label": "Insulin", "class": "Hormone"}),
        ("warfarin", {"type": "drug", "label": "Warfarin", "class": "Anticoagulant"}),
        ("aspirin", {"type": "drug", "label": "Aspirin", "class": "NSAID/Antiplatelet"}),
        ("lisinopril", {"type": "drug", "label": "Lisinopril", "class": "ACE Inhibitor"}),
        ("amlodipine", {"type": "drug", "label": "Amlodipine", "class": "CCB"}),
        ("atorvastatin", {"type": "drug", "label": "Atorvastatin", "class": "Statin"}),
        ("amoxicillin", {"type": "drug", "label": "Amoxicillin", "class": "Antibiotic"}),
        ("levothyroxine", {"type": "drug", "label": "Levothyroxine", "class": "Thyroid hormone"}),
        ("omeprazole", {"type": "drug", "label": "Omeprazole", "class": "PPI"}),

        # Lab Tests
        ("hba1c", {"type": "lab_test", "label": "HbA1c", "unit": "%", "normal": "< 5.7%"}),
        ("fasting_glucose", {"type": "lab_test", "label": "Fasting Glucose", "unit": "mg/dL", "normal": "70-100"}),
        ("creatinine", {"type": "lab_test", "label": "Serum Creatinine", "unit": "mg/dL", "normal": "0.7-1.3"}),
        ("egfr", {"type": "lab_test", "label": "eGFR", "unit": "mL/min/1.73m²", "normal": "> 60"}),
        ("inr", {"type": "lab_test", "label": "INR", "unit": "ratio", "normal": "0.8-1.2"}),
        ("wbc", {"type": "lab_test", "label": "WBC Count", "unit": "K/uL", "normal": "4.5-11.0"}),
        ("hemoglobin", {"type": "lab_test", "label": "Hemoglobin", "unit": "g/dL", "normal": "13.5-17.5"}),
        ("tsh", {"type": "lab_test", "label": "TSH", "unit": "mIU/L", "normal": "0.4-4.0"}),
        ("ldl", {"type": "lab_test", "label": "LDL Cholesterol", "unit": "mg/dL", "normal": "< 100"}),
        ("potassium", {"type": "lab_test", "label": "Serum Potassium", "unit": "meq/L", "normal": "3.5-5.1"}),
        ("blood_pressure", {"type": "lab_test", "label": "Blood Pressure", "unit": "mmHg", "normal": "< 120/80"}),

        # Procedures
        ("ecg", {"type": "procedure", "label": "Electrocardiogram (ECG)"}),
        ("chest_xray", {"type": "procedure", "label": "Chest X-Ray"}),
        ("eye_exam", {"type": "procedure", "label": "Ophthalmology Eye Exam"}),
        ("kidney_function_test", {"type": "procedure", "label": "Kidney Function Tests"}),

        # Anatomy
        ("kidney", {"type": "anatomy", "label": "Kidney"}),
        ("heart", {"type": "anatomy", "label": "Heart"}),
        ("lung", {"type": "anatomy", "label": "Lung"}),
        ("eye", {"type": "anatomy", "label": "Eye / Retina"}),
        ("liver", {"type": "anatomy", "label": "Liver"}),

        # Risk Factors
        ("obesity", {"type": "risk_factor", "label": "Obesity"}),
        ("smoking", {"type": "risk_factor", "label": "Smoking"}),
        ("sedentary_lifestyle", {"type": "risk_factor", "label": "Sedentary Lifestyle"}),
        ("family_history_diabetes", {"type": "risk_factor", "label": "Family History of Diabetes"}),
        ("high_sodium_diet", {"type": "risk_factor", "label": "High Sodium Diet"}),
    ]

    G.add_nodes_from(nodes)

    # ── Add all edges (relationships) ─────────────────────────
    edges = [
        # ── Diabetes relationships ─────────────────────────────
        ("polyuria", "type_2_diabetes", {"relation": "INDICATES", "weight": 0.8}),
        ("polydipsia", "type_2_diabetes", {"relation": "INDICATES", "weight": 0.8}),
        ("fatigue", "type_2_diabetes", {"relation": "INDICATES", "weight": 0.5}),
        ("type_2_diabetes", "metformin", {"relation": "TREATED_BY", "weight": 1.0, "first_line": True}),
        ("type_2_diabetes", "insulin", {"relation": "TREATED_BY", "weight": 0.9}),
        ("type_2_diabetes", "hba1c", {"relation": "DIAGNOSED_BY", "weight": 1.0}),
        ("type_2_diabetes", "fasting_glucose", {"relation": "DIAGNOSED_BY", "weight": 1.0}),
        ("type_2_diabetes", "diabetic_nephropathy", {"relation": "CAUSES", "weight": 0.7}),
        ("type_2_diabetes", "diabetic_retinopathy", {"relation": "CAUSES", "weight": 0.7}),
        ("type_2_diabetes", "diabetic_neuropathy", {"relation": "CAUSES", "weight": 0.7}),
        ("type_2_diabetes", "kidney", {"relation": "AFFECTS", "weight": 0.8}),
        ("type_2_diabetes", "eye", {"relation": "AFFECTS", "weight": 0.8}),
        ("type_2_diabetes", "heart", {"relation": "AFFECTS", "weight": 0.7}),
        ("diabetic_nephropathy", "type_2_diabetes", {"relation": "COMPLICATION_OF", "weight": 1.0}),
        ("diabetic_retinopathy", "type_2_diabetes", {"relation": "COMPLICATION_OF", "weight": 1.0}),
        ("obesity", "type_2_diabetes", {"relation": "RISK_FACTOR_FOR", "weight": 0.9}),
        ("sedentary_lifestyle", "type_2_diabetes", {"relation": "RISK_FACTOR_FOR", "weight": 0.7}),
        ("family_history_diabetes", "type_2_diabetes", {"relation": "RISK_FACTOR_FOR", "weight": 0.8}),
        ("metformin", "kidney", {"relation": "MONITORED_BY", "weight": 0.9}),
        ("metformin", "egfr", {"relation": "MONITORED_BY", "weight": 1.0}),
        ("metformin", "ckd", {"relation": "CONTRAINDICATED_IN", "weight": 1.0}),
        ("metformin", "lactic_acidosis", {"relation": "CAUSES", "weight": 0.2}),
        ("type_2_diabetes", "eye_exam", {"relation": "REQUIRES", "weight": 0.9}),
        ("type_2_diabetes", "kidney_function_test", {"relation": "REQUIRES", "weight": 0.9}),
        ("hba1c", "type_2_diabetes", {"relation": "MONITORED_BY", "weight": 1.0}),

        # ── Hypertension relationships ─────────────────────────
        ("headache", "hypertension", {"relation": "INDICATES", "weight": 0.4}),
        ("dizziness", "hypertension", {"relation": "INDICATES", "weight": 0.4}),
        ("hypertension", "lisinopril", {"relation": "TREATED_BY", "weight": 1.0}),
        ("hypertension", "amlodipine", {"relation": "TREATED_BY", "weight": 0.9}),
        ("hypertension", "blood_pressure", {"relation": "DIAGNOSED_BY", "weight": 1.0}),
        ("hypertension", "heart_failure", {"relation": "CAUSES", "weight": 0.6}),
        ("hypertension", "myocardial_infarction", {"relation": "RISK_FACTOR_FOR", "weight": 0.7}),
        ("hypertension", "kidney", {"relation": "AFFECTS", "weight": 0.7}),
        ("hypertension", "heart", {"relation": "AFFECTS", "weight": 0.8}),
        ("obesity", "hypertension", {"relation": "RISK_FACTOR_FOR", "weight": 0.8}),
        ("high_sodium_diet", "hypertension", {"relation": "RISK_FACTOR_FOR", "weight": 0.7}),
        ("smoking", "hypertension", {"relation": "RISK_FACTOR_FOR", "weight": 0.6}),
        ("lisinopril", "potassium", {"relation": "MONITORED_BY", "weight": 0.9}),
        ("lisinopril", "creatinine", {"relation": "MONITORED_BY", "weight": 0.9}),

        # ── Pneumonia relationships ────────────────────────────
        ("fever", "pneumonia", {"relation": "INDICATES", "weight": 0.7}),
        ("cough", "pneumonia", {"relation": "INDICATES", "weight": 0.7}),
        ("dyspnea", "pneumonia", {"relation": "INDICATES", "weight": 0.8}),
        ("chest_pain", "pneumonia", {"relation": "INDICATES", "weight": 0.5}),
        ("pneumonia", "amoxicillin", {"relation": "TREATED_BY", "weight": 0.9}),
        ("pneumonia", "wbc", {"relation": "DIAGNOSED_BY", "weight": 0.9}),
        ("pneumonia", "chest_xray", {"relation": "DIAGNOSED_BY", "weight": 1.0}),
        ("pneumonia", "lung", {"relation": "AFFECTS", "weight": 1.0}),

        # ── Cardiac relationships ──────────────────────────────
        ("chest_pain", "myocardial_infarction", {"relation": "INDICATES", "weight": 0.8}),
        ("dyspnea", "heart_failure", {"relation": "INDICATES", "weight": 0.7}),
        ("edema", "heart_failure", {"relation": "INDICATES", "weight": 0.8}),
        ("myocardial_infarction", "aspirin", {"relation": "TREATED_BY", "weight": 1.0}),
        ("myocardial_infarction", "ecg", {"relation": "DIAGNOSED_BY", "weight": 1.0}),
        ("warfarin", "inr", {"relation": "MONITORED_BY", "weight": 1.0}),
        ("aspirin", "warfarin", {"relation": "INTERACTS_WITH",
                                 "severity": "major", "weight": 1.0}),

        # ── Anemia relationships ───────────────────────────────
        ("fatigue", "anemia", {"relation": "INDICATES", "weight": 0.6}),
        ("pallor", "anemia", {"relation": "INDICATES", "weight": 0.9}),
        ("dyspnea", "anemia", {"relation": "INDICATES", "weight": 0.5}),
        ("anemia", "hemoglobin", {"relation": "DIAGNOSED_BY", "weight": 1.0}),

        # ── Thyroid relationships ──────────────────────────────
        ("fatigue", "hypothyroidism", {"relation": "INDICATES", "weight": 0.5}),
        ("weight_gain", "hypothyroidism", {"relation": "INDICATES", "weight": 0.7}),
        ("cold_intolerance", "hypothyroidism", {"relation": "INDICATES", "weight": 0.9}),
        ("hypothyroidism", "levothyroxine", {"relation": "TREATED_BY", "weight": 1.0}),
        ("hypothyroidism", "tsh", {"relation": "DIAGNOSED_BY", "weight": 1.0}),

        # ── Cholesterol relationships ──────────────────────────
        ("atorvastatin", "ldl", {"relation": "MONITORED_BY", "weight": 0.9}),
        ("atorvastatin", "liver", {"relation": "MONITORED_BY", "weight": 0.7}),
    ]

    G.add_edges_from(
        [(s, t, d) for s, t, d in edges]
    )

    logger.info(
        f"Foundation graph built | "
        f"nodes={G.number_of_nodes()} | "
        f"edges={G.number_of_edges()}"
    )
    return G