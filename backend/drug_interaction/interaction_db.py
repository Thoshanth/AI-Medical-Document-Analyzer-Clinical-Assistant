from backend.logger import get_logger

logger = get_logger("drug_interaction.db")

# ── Curated Drug Interaction Database ────────────────────────────
# Format: frozenset({drug_a, drug_b}): interaction_info
# Using frozenset so order doesn't matter (A+B == B+A)
INTERACTION_DATABASE = {
    # ── MAJOR interactions ──────────────────────────────────────
    frozenset({"warfarin", "aspirin"}): {
        "severity": "major",
        "mechanism": "Pharmacodynamic — additive anticoagulant/antiplatelet effects",
        "effect": "Significantly increased risk of serious bleeding including GI hemorrhage and intracranial bleeding",
        "management": "Avoid combination if possible. If necessary, use lowest effective aspirin dose and monitor INR closely. Educate patient on bleeding signs.",
        "clinical_significance": "This combination increases bleeding risk by 3-5x compared to warfarin alone.",
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "major",
        "mechanism": "Pharmacokinetic + pharmacodynamic — NSAIDs displace warfarin from protein binding and inhibit platelets",
        "effect": "Markedly increased anticoagulant effect and bleeding risk. Also risk of GI ulceration.",
        "management": "Avoid NSAIDs with warfarin. Use paracetamol/acetaminophen for pain instead. If unavoidable, use lowest dose for shortest duration.",
        "clinical_significance": "One of the most common causes of warfarin-related bleeding hospitalizations.",
    },
    frozenset({"metformin", "contrast dye"}): {
        "severity": "major",
        "mechanism": "Contrast media can cause acute kidney injury, reducing metformin clearance leading to accumulation",
        "effect": "Risk of metformin-induced lactic acidosis — a rare but potentially fatal condition",
        "management": "Hold metformin 48 hours before and after iodinated contrast administration. Check renal function before restarting.",
        "clinical_significance": "Standard protocol in radiology — metformin must be held for contrast procedures.",
    },
    frozenset({"atorvastatin", "clarithromycin"}): {
        "severity": "major",
        "mechanism": "Clarithromycin inhibits CYP3A4, the primary enzyme metabolizing atorvastatin",
        "effect": "Up to 10-fold increase in atorvastatin plasma levels → severe risk of rhabdomyolysis and acute kidney injury",
        "management": "Suspend atorvastatin during clarithromycin therapy. Use azithromycin as antibiotic alternative if possible.",
        "clinical_significance": "Can cause life-threatening muscle breakdown requiring dialysis.",
    },
    frozenset({"simvastatin", "clarithromycin"}): {
        "severity": "major",
        "mechanism": "CYP3A4 inhibition by clarithromycin dramatically increases simvastatin levels",
        "effect": "High risk of rhabdomyolysis and myopathy",
        "management": "Contraindicated — do not use together. Switch to pravastatin or rosuvastatin (not CYP3A4 metabolized).",
        "clinical_significance": "FDA has issued specific warnings about this combination.",
    },
    frozenset({"lisinopril", "potassium"}): {
        "severity": "major",
        "mechanism": "ACE inhibitors reduce potassium excretion; potassium supplements add more",
        "effect": "Severe hyperkalemia — can cause life-threatening cardiac arrhythmias",
        "management": "Avoid routine potassium supplementation with ACE inhibitors. Monitor serum potassium closely if supplementation necessary.",
        "clinical_significance": "Hyperkalemia from this combination can cause sudden cardiac arrest.",
    },
    frozenset({"ssri", "tramadol"}): {
        "severity": "major",
        "mechanism": "Both increase serotonin — additive serotonergic effect",
        "effect": "Serotonin syndrome — potentially life-threatening: hyperthermia, agitation, tremors, seizures",
        "management": "Avoid combination. If pain management needed, consider alternative analgesics. Monitor for serotonin syndrome symptoms.",
        "clinical_significance": "Serotonin syndrome can be rapidly fatal if not recognized and treated.",
    },
    frozenset({"ciprofloxacin", "warfarin"}): {
        "severity": "major",
        "mechanism": "Ciprofloxacin inhibits CYP1A2 and alters gut flora affecting vitamin K synthesis",
        "effect": "Significantly potentiates warfarin anticoagulation — major bleeding risk",
        "management": "Monitor INR very closely during and after ciprofloxacin therapy. Anticipate INR increase and consider prophylactic dose reduction.",
        "clinical_significance": "INR can double or triple within days of starting ciprofloxacin.",
    },
    frozenset({"methotrexate", "nsaid"}): {
        "severity": "major",
        "mechanism": "NSAIDs reduce renal methotrexate clearance causing dangerous accumulation",
        "effect": "Methotrexate toxicity — bone marrow suppression, mucositis, hepatotoxicity, nephrotoxicity",
        "management": "Avoid NSAIDs with methotrexate. Use paracetamol for pain. If low-dose methotrexate for arthritis, short NSAID courses may be acceptable with close monitoring.",
        "clinical_significance": "Can be fatal. This is a well-documented cause of methotrexate toxicity deaths.",
    },
    frozenset({"digoxin", "amiodarone"}): {
        "severity": "major",
        "mechanism": "Amiodarone inhibits P-glycoprotein and reduces digoxin renal clearance",
        "effect": "Digoxin toxicity — nausea, visual disturbances, life-threatening arrhythmias",
        "management": "Reduce digoxin dose by 50% when starting amiodarone. Monitor digoxin levels closely.",
        "clinical_significance": "Digoxin has a very narrow therapeutic window — small increases cause toxicity.",
    },

    # ── MODERATE interactions ───────────────────────────────────
    frozenset({"metformin", "alcohol"}): {
        "severity": "moderate",
        "mechanism": "Alcohol inhibits gluconeogenesis and increases risk of lactic acidosis with metformin",
        "effect": "Increased risk of hypoglycemia and lactic acidosis, especially with heavy alcohol use",
        "management": "Advise patient to limit alcohol consumption. Avoid binge drinking completely.",
        "clinical_significance": "Regular moderate alcohol use significantly increases metformin side effect risk.",
    },
    frozenset({"amlodipine", "simvastatin"}): {
        "severity": "moderate",
        "mechanism": "Amlodipine is a weak CYP3A4 inhibitor, slightly increasing simvastatin levels",
        "effect": "Increased risk of myopathy and rhabdomyolysis",
        "management": "Do not exceed simvastatin 20mg daily when combined with amlodipine. Consider switching to atorvastatin.",
        "clinical_significance": "FDA limits simvastatin dose to 20mg with amlodipine.",
    },
    frozenset({"lisinopril", "ibuprofen"}): {
        "severity": "moderate",
        "mechanism": "NSAIDs antagonize the antihypertensive effect and reduce renal protection of ACE inhibitors",
        "effect": "Reduced blood pressure control, increased risk of acute kidney injury, especially in elderly or dehydrated patients",
        "management": "Use paracetamol instead. If NSAID necessary, use lowest dose, shortest duration, monitor blood pressure and renal function.",
        "clinical_significance": "This triple whammy with diuretics causes significant AKI hospitalizations.",
    },
    frozenset({"omeprazole", "clopidogrel"}): {
        "severity": "moderate",
        "mechanism": "Omeprazole inhibits CYP2C19 which activates clopidogrel to its active form",
        "effect": "Reduced antiplatelet effect of clopidogrel — may increase cardiovascular events",
        "management": "Use pantoprazole instead of omeprazole (lower CYP2C19 inhibition). FDA recommends avoiding omeprazole with clopidogrel.",
        "clinical_significance": "Important in patients with coronary stents — reduced antiplatelet effect increases stent thrombosis risk.",
    },
    frozenset({"insulin", "beta-blocker"}): {
        "severity": "moderate",
        "mechanism": "Beta-blockers mask tachycardia (the main warning sign of hypoglycemia) and can prolong hypoglycemia",
        "effect": "Hypoglycemia may be prolonged and signs may be masked (sweating still present, but heart racing masked)",
        "management": "Use cardioselective beta-blockers (metoprolol, atenolol) when possible. Educate patient to monitor glucose more frequently.",
        "clinical_significance": "Patient may not notice hypoglycemia until severely low.",
    },

    # ── MINOR interactions ──────────────────────────────────────
    frozenset({"omeprazole", "iron"}): {
        "severity": "minor",
        "mechanism": "Omeprazole raises gastric pH, reducing iron absorption (iron absorption requires acidic environment)",
        "effect": "Reduced iron supplement absorption — may reduce efficacy of iron replacement therapy",
        "management": "Take iron 2 hours before or 4 hours after omeprazole. Monitor iron stores.",
        "clinical_significance": "Clinically relevant in patients with iron deficiency anemia on PPIs.",
    },
    frozenset({"metformin", "vitamin b12"}): {
        "severity": "minor",
        "mechanism": "Metformin reduces ileal absorption of vitamin B12",
        "effect": "Gradual vitamin B12 depletion over months to years — may cause peripheral neuropathy",
        "management": "Monitor B12 levels annually in patients on long-term metformin. Supplement if deficient.",
        "clinical_significance": "Up to 30% of metformin users develop B12 deficiency over time.",
    },
}


def check_pair_in_db(
    drug_a: str,
    drug_b: str,
) -> dict | None:
    """
    Checks two drugs against the curated interaction database.
    Case-insensitive, partial name matching.

    Returns interaction dict if found, None otherwise.
    """
    name_a = drug_a.lower().strip()
    name_b = drug_b.lower().strip()

    # Exact match first
    key = frozenset({name_a, name_b})
    if key in INTERACTION_DATABASE:
        logger.info(
            f"Interaction found (exact) | "
            f"'{drug_a}' + '{drug_b}'"
        )
        return INTERACTION_DATABASE[key]

    # Partial match — check if either drug name contains a key drug name
    for db_key, interaction in INTERACTION_DATABASE.items():
        db_drugs = list(db_key)
        match_a = any(d in name_a or name_a in d for d in db_drugs)
        match_b = any(d in name_b or name_b in d for d in db_drugs)

        if match_a and match_b:
            logger.info(
                f"Interaction found (partial) | "
                f"'{drug_a}' + '{drug_b}'"
            )
            return interaction

    return None


def get_all_interactions_for_drug(drug_name: str) -> list[dict]:
    """
    Returns all known interactions for a specific drug.
    Useful for building a complete drug profile.
    """
    name_lower = drug_name.lower()
    interactions = []

    for key, interaction in INTERACTION_DATABASE.items():
        drugs = list(key)
        if any(d in name_lower or name_lower in d for d in drugs):
            other_drug = next(
                d for d in drugs
                if d not in name_lower and name_lower not in d
            )
            interactions.append({
                "interacts_with": other_drug,
                **interaction,
            })

    return interactions