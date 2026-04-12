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
from backend.medical_rag.rag_pipeline import (
    index_medical_document,
    medical_rag_query,
)
from backend.ingestion.extractor import extract_text
from backend.clinical_nlp.nlp_pipeline import run_clinical_nlp, get_clinical_entities
from backend.knowledge_base.kb_pipeline import enrich_with_knowledge_base
from backend.knowledge_base.drug_db import get_drug_info
from backend.knowledge_base.disease_db import get_disease_info, get_all_diseases
from backend.drug_interaction.interaction_pipeline import (
    check_all_medications,
    check_drug_pair,
)
from backend.ingestion.cleaner import clean_medical_text, get_word_count
from backend.medical_graph.graph_pipeline import (
    build_medical_graph,
    query_medical_graph,
    get_patient_summary,
    explore_graph,
)
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

# ══════════════════════════════════════════════════════════════════
# STAGE 2 — Clinical NLP Pipeline
# ══════════════════════════════════════════════════════════════════

@app.post("/analyze/{document_id}", tags=["Stage 2 - Clinical NLP"])
def analyze_document(document_id: int):
    """
    Run full clinical NLP on an uploaded document.

    Extracts:
    - Symptoms (with severity and duration)
    - Diagnoses (with ICD-10 codes)
    - Medications (normalized with drug class)
    - Lab values (interpreted with normal ranges)
    - Vital signs
    - Procedures

    Also generates clinical alerts for critical findings.
    """
    logger.info(f"Clinical NLP requested | doc_id={document_id}")
    try:
        result = run_clinical_nlp(document_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Clinical NLP failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/analyze/{document_id}", tags=["Stage 2 - Clinical NLP"])
def get_analysis(document_id: int):
    """
    Get previously stored clinical NLP results.
    No LLM call — reads from database.
    """
    result = get_clinical_entities(document_id)
    if not result:
        raise HTTPException(
            404,
            f"No analysis found for document {document_id}. "
            f"Run POST /analyze/{document_id} first."
        )
    return result
# ══════════════════════════════════════════════════════════════════
# STAGE 3 — Medical Knowledge Base
# ══════════════════════════════════════════════════════════════════

@app.post("/knowledge/enrich/{document_id}", tags=["Stage 3 - Knowledge Base"])
def enrich_document(document_id: int):
    """
    Enriches clinical entities with medical knowledge base.

    Requires Stage 2 analysis to be run first.

    Adds to each diagnosis:
    - Disease description, treatments, complications
    - Monitoring requirements, emergency signs

    Adds to each medication:
    - Drug class, indications, contraindications
    - Side effects, interactions, high-risk flag

    Also runs symptom checker and builds clinical summary.
    """
    logger.info(f"KB enrichment | doc_id={document_id}")
    try:
        result = enrich_with_knowledge_base(document_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"KB enrichment failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/knowledge/drug/{drug_name}", tags=["Stage 3 - Knowledge Base"])
def lookup_drug(drug_name: str):
    """
    Look up drug information by name.
    Searches local DB then OpenFDA API.
    """
    info = get_drug_info(drug_name)
    if not info:
        raise HTTPException(404, f"Drug '{drug_name}' not found")
    return info


@app.get("/knowledge/disease/{disease_name}", tags=["Stage 3 - Knowledge Base"])
def lookup_disease(disease_name: str):
    """
    Look up disease information by name.
    Returns full clinical profile.
    """
    info = get_disease_info(disease_name)
    if not info:
        raise HTTPException(
            404,
            f"Disease '{disease_name}' not found in knowledge base"
        )
    return info


@app.get("/knowledge/diseases", tags=["Stage 3 - Knowledge Base"])
def list_diseases():
    """List all diseases in the knowledge base."""
    return {"diseases": get_all_diseases()}


# ══════════════════════════════════════════════════════════════════
# STAGE 4 — Medical RAG Pipeline
# ══════════════════════════════════════════════════════════════════

@app.post("/medical-rag/index/{document_id}", tags=["Stage 4 - Medical RAG"])
def index_for_rag(document_id: int):
    """
    Index a medical document for RAG queries.

    Uses medical-aware chunking (preserves lab sections,
    medication lists, SOAP note sections) and PubMedBERT
    embeddings for medical domain accuracy.

    Run this after /upload and /analyze/{id}.
    """
    logger.info(f"Medical RAG index | doc_id={document_id}")
    try:
        result = index_medical_document(document_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/medical-rag/query", tags=["Stage 4 - Medical RAG"])
def medical_query(
    question: str,
    document_id: int = None,
    top_k: int = 5,
    include_kb: bool = True,
):
    """
    Ask a medical question. Returns cited, safe answer.

    Combines:
    - PubMedBERT semantic retrieval
    - BM25 keyword matching
    - Medical entity boost
    - Stage 2 clinical entity context
    - Stage 3 knowledge base enrichment

    Always includes medical disclaimer and source citations.
    Critical findings are flagged prominently.

    document_id: filter to one document (optional)
    include_kb: include knowledge base context (default True)
    """
    logger.info(
        f"Medical RAG query | "
        f"question='{question[:50]}' | "
        f"doc_id={document_id}"
    )
    try:
        return medical_rag_query(
            question=question,
            document_id=document_id,
            top_k=top_k,
            include_kb=include_kb,
        )
    except Exception as e:
        logger.error(f"Medical query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    
# ══════════════════════════════════════════════════════════════════
# STAGE 5 — Drug Interaction Checker
# ══════════════════════════════════════════════════════════════════

@app.post("/interactions/check/{document_id}", tags=["Stage 5 - Drug Interactions"])
def check_document_interactions(document_id: int):
    """
    Check all drug interactions for medications in a document.

    Requires Stage 2 analysis (POST /analyze/{id}) to be run first.

    Checks every unique medication pair using:
    1. Curated interaction database (instant)
    2. OpenFDA drug label API (authoritative)
    3. MiniMax LLM (broad pharmacological knowledge)

    Returns interaction report sorted by severity
    with clinical management recommendations.
    """
    logger.info(f"Drug interaction check | doc_id={document_id}")
    try:
        result = check_all_medications(document_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Interaction check failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/interactions/check-pair", tags=["Stage 5 - Drug Interactions"])
def check_pair_endpoint(drug_a: str, drug_b: str):
    """
    Check interaction between two specific drugs.

    Useful for quick single pair checks without uploading a document.

    Examples:
    - drug_a=warfarin, drug_b=aspirin
    - drug_a=metformin, drug_b=contrast dye
    - drug_a=atorvastatin, drug_b=clarithromycin
    """
    logger.info(f"Pair check | '{drug_a}' + '{drug_b}'")
    try:
        result = check_drug_pair(drug_a, drug_b)
        return result
    except Exception as e:
        logger.error(f"Pair check failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    
# ══════════════════════════════════════════════════════════════════
# STAGE 6 — Medical Knowledge Graph
# ══════════════════════════════════════════════════════════════════

@app.post("/medical-graph/build/{document_id}", tags=["Stage 6 - Medical Graph"])
def build_graph(document_id: int):
    """
    Builds a medical knowledge graph for a document.

    Layer 1: Pre-built foundation graph (diseases, drugs,
             symptoms, labs — all interconnected)
    Layer 2: Document-specific entities and relationships
             extracted by MiniMax + Stage 2 clinical entities

    Combined graph enables multi-hop clinical reasoning.
    Requires Stage 2 analysis to be run first.
    """
    logger.info(f"Build graph | doc_id={document_id}")
    try:
        result = build_medical_graph(document_id)
        return result
    except Exception as e:
        logger.error(f"Graph build failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/medical-graph/query", tags=["Stage 6 - Medical Graph"])
def graph_query(question: str, document_id: int):
    """
    Answer a clinical question using graph-enhanced reasoning.

    Better than /medical-rag/query for relationship questions:
    - What complications should we watch for?
    - What labs need to be monitored for these medications?
    - What conditions contraindicate this drug?
    - What is the differential diagnosis for these symptoms?
    """
    logger.info(f"Graph query | question='{question[:50]}'")
    try:
        return query_medical_graph(question, document_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Graph query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/medical-graph/patient-summary/{document_id}", tags=["Stage 6 - Medical Graph"])
def patient_summary(document_id: int):
    """
    Complete patient clinical summary from graph traversal.

    Shows:
    - Complications to watch for each condition
    - Monitoring requirements
    - Drug-condition contraindications
    - Differential diagnosis from symptoms
    """
    try:
        return get_patient_summary(document_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Patient summary failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/medical-graph/explore/{document_id}", tags=["Stage 6 - Medical Graph"])
def explore(document_id: int):
    """
    Explore the medical knowledge graph for a document.
    Returns all nodes, edges, and relationship types.
    """
    try:
        return explore_graph(document_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Explore failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))