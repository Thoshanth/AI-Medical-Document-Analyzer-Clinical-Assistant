import json
import requests
import time
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("knowledge_base.drug")

CACHE_DIR = Path("knowledge_cache")
CACHE_DIR.mkdir(exist_ok=True)
DRUG_CACHE_FILE = CACHE_DIR / "drug_cache.json"

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

# Local curated drug database for common medications
# Used as fallback when OpenFDA is unavailable
LOCAL_DRUG_DB = {
    "metformin": {
        "brand_names": ["Glucophage", "Fortamet", "Glumetza"],
        "drug_class": "Biguanide antidiabetic",
        "indications": ["Type 2 diabetes mellitus"],
        "mechanism": "Decreases hepatic glucose production, improves insulin sensitivity",
        "contraindications": [
            "severe kidney disease (eGFR < 30)",
            "liver disease",
            "alcoholism",
            "iodinated contrast dye (hold 48h before/after)"
        ],
        "side_effects": [
            "nausea", "diarrhea", "stomach upset",
            "lactic acidosis (rare but serious)", "vitamin B12 deficiency"
        ],
        "monitoring": ["kidney function", "vitamin B12 annually", "HbA1c"],
        "interactions": ["alcohol", "contrast dye", "cimetidine"],
        "pregnancy_category": "B",
        "high_risk": False,
    },
    "warfarin": {
        "brand_names": ["Coumadin", "Jantoven"],
        "drug_class": "Vitamin K antagonist anticoagulant",
        "indications": [
            "atrial fibrillation", "DVT", "pulmonary embolism",
            "mechanical heart valves"
        ],
        "mechanism": "Inhibits vitamin K-dependent clotting factors (II, VII, IX, X)",
        "contraindications": [
            "active bleeding", "pregnancy",
            "recent surgery", "severe hypertension"
        ],
        "side_effects": [
            "bleeding (major risk)", "bruising",
            "skin necrosis (rare)", "purple toe syndrome"
        ],
        "monitoring": ["INR weekly initially then monthly", "signs of bleeding"],
        "interactions": [
            "NSAIDs", "aspirin", "antibiotics",
            "vitamin K foods", "many medications"
        ],
        "pregnancy_category": "X",
        "high_risk": True,
    },
    "amoxicillin": {
        "brand_names": ["Amoxil", "Trimox"],
        "drug_class": "Penicillin antibiotic",
        "indications": [
            "respiratory infections", "ear infections",
            "urinary tract infections", "skin infections",
            "H. pylori eradication"
        ],
        "mechanism": "Inhibits bacterial cell wall synthesis",
        "contraindications": [
            "penicillin allergy", "severe renal impairment"
        ],
        "side_effects": [
            "diarrhea", "nausea", "skin rash",
            "allergic reaction", "C. difficile colitis"
        ],
        "monitoring": ["signs of allergic reaction", "renal function"],
        "interactions": ["methotrexate", "warfarin", "oral contraceptives"],
        "pregnancy_category": "B",
        "high_risk": False,
    },
    "atorvastatin": {
        "brand_names": ["Lipitor"],
        "drug_class": "HMG-CoA reductase inhibitor (Statin)",
        "indications": [
            "hypercholesterolemia", "cardiovascular disease prevention",
            "dyslipidemia"
        ],
        "mechanism": "Inhibits HMG-CoA reductase, reducing cholesterol synthesis",
        "contraindications": [
            "active liver disease", "pregnancy",
            "breastfeeding"
        ],
        "side_effects": [
            "myalgia", "rhabdomyolysis (rare)",
            "liver enzyme elevation", "diabetes risk"
        ],
        "monitoring": ["LFTs at baseline", "CK if muscle symptoms", "lipid panel"],
        "interactions": ["fibrates", "niacin", "cyclosporine", "clarithromycin"],
        "pregnancy_category": "X",
        "high_risk": False,
    },
    "lisinopril": {
        "brand_names": ["Prinivil", "Zestril"],
        "drug_class": "ACE Inhibitor",
        "indications": [
            "hypertension", "heart failure",
            "diabetic nephropathy", "post-MI"
        ],
        "mechanism": "Inhibits ACE, reducing angiotensin II and aldosterone",
        "contraindications": [
            "pregnancy", "bilateral renal artery stenosis",
            "history of angioedema", "hyperkalemia"
        ],
        "side_effects": [
            "dry cough", "hyperkalemia",
            "angioedema (rare but dangerous)", "hypotension"
        ],
        "monitoring": [
            "blood pressure", "potassium",
            "creatinine (first 2 weeks)", "symptoms of angioedema"
        ],
        "interactions": ["potassium supplements", "NSAIDs", "lithium"],
        "pregnancy_category": "D",
        "high_risk": False,
    },
    "insulin": {
        "brand_names": [
            "Humalog", "Novolog", "Lantus",
            "Levemir", "Toujeo"
        ],
        "drug_class": "Insulin (antidiabetic hormone)",
        "indications": [
            "Type 1 diabetes", "Type 2 diabetes",
            "diabetic ketoacidosis", "hyperglycemia"
        ],
        "mechanism": "Facilitates glucose uptake into cells, lowers blood sugar",
        "contraindications": ["hypoglycemia"],
        "side_effects": [
            "hypoglycemia (most common)", "weight gain",
            "lipodystrophy at injection sites",
            "hypokalemia (IV insulin)"
        ],
        "monitoring": [
            "blood glucose multiple times daily",
            "HbA1c every 3 months", "injection sites"
        ],
        "interactions": [
            "beta-blockers (mask hypoglycemia symptoms)",
            "corticosteroids", "thiazide diuretics"
        ],
        "pregnancy_category": "B",
        "high_risk": True,
    },
    "omeprazole": {
        "brand_names": ["Prilosec", "Losec"],
        "drug_class": "Proton Pump Inhibitor (PPI)",
        "indications": [
            "GERD", "peptic ulcer disease",
            "H. pylori eradication", "Zollinger-Ellison syndrome"
        ],
        "mechanism": "Inhibits H+/K+ ATPase in gastric parietal cells",
        "contraindications": ["hypersensitivity to PPIs"],
        "side_effects": [
            "headache", "diarrhea", "nausea",
            "vitamin B12 deficiency (long-term)",
            "hypomagnesemia (long-term)",
            "C. difficile (increased risk)"
        ],
        "monitoring": [
            "magnesium levels (long-term use)",
            "vitamin B12 (long-term use)"
        ],
        "interactions": [
            "clopidogrel", "methotrexate",
            "digoxin", "iron supplements"
        ],
        "pregnancy_category": "C",
        "high_risk": False,
    },
    "amlodipine": {
        "brand_names": ["Norvasc"],
        "drug_class": "Calcium Channel Blocker (dihydropyridine)",
        "indications": [
            "hypertension", "angina", "coronary artery disease"
        ],
        "mechanism": "Blocks L-type calcium channels, causing vasodilation",
        "contraindications": [
            "cardiogenic shock", "severe aortic stenosis"
        ],
        "side_effects": [
            "peripheral edema", "flushing",
            "headache", "dizziness", "palpitations"
        ],
        "monitoring": ["blood pressure", "heart rate", "edema"],
        "interactions": [
            "cyclosporine", "simvastatin (limit dose)",
            "CYP3A4 inhibitors"
        ],
        "pregnancy_category": "C",
        "high_risk": False,
    },
}


def _load_cache() -> dict:
    """Loads drug cache from disk."""
    if DRUG_CACHE_FILE.exists():
        try:
            with open(DRUG_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    """Saves drug cache to disk."""
    with open(DRUG_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_drug_info(drug_name: str) -> dict | None:
    """
    Gets drug information using 3-tier lookup:
    1. Local curated database (instant)
    2. Disk cache from previous OpenFDA calls
    3. OpenFDA API (live, then cached)
    """
    name_lower = drug_name.lower().strip()

    # Tier 1: Local database
    for db_name, info in LOCAL_DRUG_DB.items():
        if db_name in name_lower or name_lower in db_name:
            logger.info(f"Drug found (local) | '{drug_name}'")
            return {**info, "source": "local_database", "name": drug_name}

    # Tier 2: Disk cache
    cache = _load_cache()
    if name_lower in cache:
        logger.info(f"Drug found (cache) | '{drug_name}'")
        return cache[name_lower]

    # Tier 3: OpenFDA API
    logger.info(f"Querying OpenFDA | '{drug_name}'")
    try:
        response = requests.get(
            OPENFDA_BASE,
            params={
                "search": f"openfda.generic_name:{drug_name}",
                "limit": 1,
            },
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if results:
                result = results[0]
                drug_info = _parse_fda_result(result, drug_name)

                # Cache for future use
                cache[name_lower] = drug_info
                _save_cache(cache)

                logger.info(f"Drug found (OpenFDA) | '{drug_name}'")
                return drug_info

    except requests.RequestException as e:
        logger.warning(f"OpenFDA API failed: {e}")

    logger.debug(f"Drug not found | '{drug_name}'")
    return None


def _parse_fda_result(result: dict, drug_name: str) -> dict:
    """Parses OpenFDA API response into our standard format."""

    def get_first(key: str) -> str:
        val = result.get(key, [""])[0] if result.get(key) else ""
        return val[:500] if val else ""

    openfda = result.get("openfda", {})

    return {
        "name": drug_name,
        "brand_names": openfda.get("brand_name", [])[:3],
        "drug_class": openfda.get("pharm_class_epc", ["Unknown"])[0]
        if openfda.get("pharm_class_epc") else "Unknown",
        "indications": [get_first("indications_and_usage")],
        "contraindications": [get_first("contraindications")],
        "side_effects": [get_first("adverse_reactions")],
        "warnings": [get_first("warnings")],
        "interactions": [get_first("drug_interactions")],
        "dosage": get_first("dosage_and_administration"),
        "source": "openfda",
        "high_risk": False,
    }