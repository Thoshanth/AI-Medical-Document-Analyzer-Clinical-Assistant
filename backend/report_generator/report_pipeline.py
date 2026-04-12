import json
from pathlib import Path
from datetime import datetime
from backend.report_generator.soap_generator import generate_soap_note
from backend.report_generator.differential_generator import (
    generate_differential_report,
)
from backend.report_generator.medication_report import (
    generate_medication_report,
)
from backend.report_generator.lab_report_generator import generate_lab_report
from backend.logger import get_logger

logger = get_logger("report_generator.pipeline")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def generate_full_report(document_id: int) -> dict:
    """
    Generates all four report types and combines them
    into a comprehensive clinical document.

    Runs each report generator and combines results.
    Failed individual reports don't fail the whole report.
    """
    logger.info(f"Generating full clinical report | doc_id={document_id}")

    results = {
        "document_id": document_id,
        "generated_at": datetime.utcnow().isoformat(),
        "reports": {},
        "errors": {},
    }

    # Generate each report type
    report_generators = {
        "soap_note": generate_soap_note,
        "differential_diagnosis": generate_differential_report,
        "medication_review": generate_medication_report,
        "lab_interpretation": generate_lab_report,
    }

    for report_type, generator in report_generators.items():
        try:
            logger.info(f"Generating {report_type}")
            results["reports"][report_type] = generator(document_id)
            logger.info(f"{report_type} complete")
        except Exception as e:
            logger.error(f"{report_type} failed: {e}")
            results["errors"][report_type] = str(e)

    # Save to disk
    report_path = REPORTS_DIR / f"report_doc_{document_id}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full report saved | path='{report_path}'")

    results["reports_generated"] = len(results["reports"])
    results["reports_failed"] = len(results["errors"])
    results["saved_to"] = str(report_path)

    return results


def get_saved_report(document_id: int) -> dict | None:
    """Retrieves a previously generated report from disk."""
    report_path = REPORTS_DIR / f"report_doc_{document_id}.json"
    if not report_path.exists():
        return None
    with open(report_path, "r") as f:
        return json.load(f)