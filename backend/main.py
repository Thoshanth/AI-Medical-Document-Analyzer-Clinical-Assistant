import os
import json
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from backend.database.db import init_db
from backend.ingestion.extractor import extract_text
from backend.ingestion.cleaner import clean_medical_text, get_word_count
from backend.ingestion.classifier import classify_document
from backend.ingestion.medical_metadata import extract_medical_metadata
from backend.storage.document_store import (
    save_medical_document,
    get_all_documents,
    get_document_by_id,
)
from backend.logger import get_logger

logger = get_logger("main")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"pdf", "txt", "json", "csv"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AI Medical Platform starting")
    logger.info("=" * 60)
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("AI Medical Platform shutting down")


app = FastAPI(
    title="AI Medical Document Analyzer",
    version="0.1.0",
    docs_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AI Medical Platform",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — Medical Document Ingestion
# ══════════════════════════════════════════════════════════════════

@app.post("/upload", tags=["Stage 1 - Ingestion"])
async def upload_medical_document(file: UploadFile = File(...)):
    """
    Upload any medical document: PDF, TXT, JSON (FHIR), CSV.

    Pipeline:
    1. Extract text (smart routing by file type)
    2. Classify document type (lab report, prescription, etc.)
    3. Clean and normalize medical text
    4. Extract medical metadata (urgency, PII, facility)
    5. Store everything in SQLite
    """
    logger.info(f"Upload received | file='{file.filename}'")
    start = time.time()

    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_TYPES:
        raise HTTPException(
            400,
            f"Unsupported file type '{extension}'. "
            f"Allowed: {list(ALLOWED_TYPES)}"
        )

    # Save to disk
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
    logger.info(f"File saved | size={file_size_kb}KB")

    # Extract text
    raw_text, page_count = extract_text(str(file_path), extension)

    # Classify document type
    doc_type = classify_document(raw_text, file.filename)
    logger.info(f"Document classified | type={doc_type}")

    # Clean text (expand abbreviations for clinical docs)
    expand_abbrev = doc_type in ["clinical_note", "prescription", "lab_report"]
    clean_text = clean_medical_text(raw_text, expand_abbreviations=expand_abbrev)
    word_count = get_word_count(clean_text)

    # Extract medical metadata
    metadata = extract_medical_metadata(clean_text, doc_type)

    # Save to database
    doc_id = save_medical_document(
        filename=file.filename,
        file_type=extension,
        document_type=doc_type,
        page_count=page_count,
        word_count=word_count,
        file_size_kb=file_size_kb,
        extracted_text=clean_text,
        medical_metadata=metadata,
    )

    elapsed = round(time.time() - start, 3)
    logger.info(
        f"Upload complete | id={doc_id} | "
        f"type={doc_type} | urgency={metadata['urgency_level']} | "
        f"time={elapsed}s"
    )

    # Build response
    response = {
        "message": "Medical document processed successfully",
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": extension,
        "document_type": doc_type,
        "page_count": page_count,
        "word_count": word_count,
        "file_size_kb": file_size_kb,
        "processing_time_seconds": elapsed,
        "medical_info": {
            "urgency_level": metadata["urgency_level"],
            "has_pii": metadata["has_pii"],
            "pii_types_found": metadata["pii_types_found"],
            "medical_facility": metadata["medical_facility"],
            "document_date": metadata["document_date"],
        },
    }

    # Add emergency warning to response
    if metadata["urgency_level"] == "emergency":
        response["⚠️ EMERGENCY_ALERT"] = (
            "This document contains emergency indicators. "
            "Immediate medical attention may be required."
        )
        logger.warning(f"EMERGENCY document uploaded | id={doc_id}")

    return response


@app.get("/documents", tags=["Stage 1 - Ingestion"])
def list_documents():
    """List all uploaded medical documents with metadata."""
    return get_all_documents()


@app.get("/documents/{document_id}", tags=["Stage 1 - Ingestion"])
def get_document(document_id: int):
    """Get a specific document's details."""
    record = get_document_by_id(document_id)
    if not record:
        raise HTTPException(404, f"Document {document_id} not found")
    return {
        "id": record.id,
        "filename": record.filename,
        "document_type": record.document_type,
        "urgency_level": record.urgency_level,
        "has_pii": record.has_pii,
        "word_count": record.word_count,
        "medical_metadata": json.loads(record.medical_metadata or "{}"),
        "text_preview": record.extracted_text[:500] + "...",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "service": "AI Medical Platform", "version": "0.1.0"}