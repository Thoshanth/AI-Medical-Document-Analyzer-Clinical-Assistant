import fitz
import pdfplumber
import time
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("ingestion.extractor")


def extract_text(file_path: str, file_type: str) -> tuple[str, int | None]:
    """
    Routes extraction based on file type.
    Medical-specific handling for each format.
    """
    logger.info(
        f"Extraction starting | file='{file_path}' | type='{file_type}'"
    )
    start = time.time()

    if file_type == "pdf":
        result = _extract_pdf(file_path)
    elif file_type == "json":
        result = _extract_fhir(file_path)
    elif file_type == "txt":
        result = _extract_txt(file_path)
    elif file_type == "csv":
        result = _extract_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    elapsed = round(time.time() - start, 3)
    text, pages = result
    logger.info(
        f"Extraction complete | chars={len(text)} | "
        f"pages={pages} | time={elapsed}s"
    )
    return result


def _extract_pdf(file_path: str) -> tuple[str, int]:
    """
    Medical PDF extraction with scanned detection.
    Medical PDFs often have complex layouts — tables, forms, multi-column.
    """
    text = ""
    page_count = 0

    # Try PyMuPDF first
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            text += page_text
            logger.debug(f"PDF page {i+1}/{page_count} | chars={len(page_text)}")
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")

    # Fallback to pdfplumber (better for medical tables/forms)
    if len(text.strip()) < 100:
        logger.warning(
            "Insufficient text from PyMuPDF — trying pdfplumber"
        )
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    # Extract tables as text too
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text += " | ".join(
                                    str(cell) for cell in row if cell
                                ) + "\n"

                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            logger.info("pdfplumber extraction succeeded")
        except Exception as e:
            logger.error(f"pdfplumber failed: {e}", exc_info=True)
            raise

    # Check if still empty — likely scanned
    if len(text.strip()) < 50:
        logger.warning(
            "PDF appears to be scanned — flagging for vision processing"
        )
        text = "[SCANNED_DOCUMENT: This PDF contains images only. " \
               "Vision processing recommended.]"

    return text, page_count


def _extract_fhir(file_path: str) -> tuple[str, None]:
    """Routes FHIR JSON to the FHIR parser."""
    from backend.ingestion.fhir_parser import parse_fhir_json
    text = parse_fhir_json(file_path)
    return text, None


def _extract_txt(file_path: str) -> tuple[str, None]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), None
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed — trying latin-1")
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read(), None


def _extract_csv(file_path: str) -> tuple[str, None]:
    import pandas as pd
    df = pd.read_csv(file_path)
    text = f"Columns: {', '.join(df.columns.tolist())}\n\n"
    text += f"Total rows: {len(df)}\n\n"
    text += "Data:\n"
    text += df.to_string(index=False)
    logger.info(f"CSV extracted | rows={len(df)} | cols={len(df.columns)}")
    return text, None