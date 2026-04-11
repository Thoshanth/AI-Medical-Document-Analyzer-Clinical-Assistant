import json
import numpy as np
import chromadb
from pathlib import Path
from backend.medical_rag.medical_chunker import chunk_medical_document
from backend.medical_rag.medical_embedder import embed_texts, embed_query
from backend.medical_rag.medical_retriever import hybrid_medical_retrieval
from backend.medical_rag.answer_generator import generate_medical_answer
from backend.clinical_nlp.nlp_pipeline import get_clinical_entities
from backend.knowledge_base.kb_pipeline import enrich_with_knowledge_base
from backend.database.db import SessionLocal, MedicalDocument
from backend.logger import get_logger

logger = get_logger("medical_rag.pipeline")

# Separate ChromaDB collection for medical RAG
CHROMA_PATH = Path("chroma_db")
CHROMA_PATH.mkdir(exist_ok=True)
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_medical_collection():
    """Gets or creates the medical documents ChromaDB collection."""
    return _chroma_client.get_or_create_collection(
        name="medical_documents",
        metadata={"hnsw:space": "cosine"},
    )


def index_medical_document(
    document_id: int,
) -> dict:
    """
    Medical indexing pipeline:
    1. Load document from SQLite (Stage 1)
    2. Medical-aware chunking (section/paragraph)
    3. PubMedBERT embedding
    4. Store in ChromaDB with rich medical metadata

    Returns indexing summary.
    """
    logger.info(f"Medical indexing | doc_id={document_id}")

    # Load document
    db = SessionLocal()
    try:
        record = db.query(MedicalDocument).filter(
            MedicalDocument.id == document_id
        ).first()
        if not record:
            raise ValueError(f"Document {document_id} not found")
        text = record.extracted_text
        doc_type = record.document_type
        filename = record.filename
    finally:
        db.close()

    # Medical-aware chunking
    chunks = chunk_medical_document(text, doc_type)
    logger.info(f"Chunks created | count={len(chunks)}")

    if not chunks:
        raise ValueError("No chunks created — document may be empty")

    # Embed with PubMedBERT
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts)

    # Store in ChromaDB
    collection = get_medical_collection()

    ids = [
        f"doc{document_id}_chunk{i}_{doc_type}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "doc_type": doc_type,
            "section": chunk.get("section", "unknown"),
            "chunk_type": chunk.get("chunk_type", "unknown"),
            "chunk_index": i,
            "word_count": len(chunk["text"].split()),
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunk_texts,
        metadatas=metadatas,
    )

    logger.info(
        f"Medical indexing complete | "
        f"doc_id={document_id} | "
        f"chunks={len(chunks)} | "
        f"collection_total={collection.count()}"
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "document_type": doc_type,
        "chunks_created": len(chunks),
        "chunks_by_type": {
            chunk.get("chunk_type", "unknown"): sum(
                1 for c in chunks
                if c.get("chunk_type") == chunk.get("chunk_type")
            )
            for chunk in chunks
        },
        "status": "indexed",
        "collection_total": collection.count(),
    }


def medical_rag_query(
    question: str,
    document_id: int = None,
    top_k: int = 5,
    include_kb: bool = True,
) -> dict:
    """
    Full medical RAG query pipeline:
    1. Embed question with PubMedBERT
    2. Retrieve from ChromaDB
    3. Hybrid reranking (semantic + BM25 + medical boost)
    4. Load clinical entity context (Stage 2)
    5. Load KB context (Stage 3) if available
    6. Generate safe medical answer with citations
    7. Return answer + sources + critical alerts + disclaimer
    """
    logger.info(
        f"Medical RAG query | "
        f"question='{question[:60]}' | "
        f"doc_id={document_id} | top_k={top_k}"
    )

    collection = get_medical_collection()

    if collection.count() == 0:
        return {
            "answer": "No documents have been indexed yet. "
                      "Please upload and index a document first.",
            "sources": [],
            "critical_alerts": [],
            "disclaimer": "",
        }

    # Step 1: Embed query
    query_embedding = embed_query(question)

    # Step 2: ChromaDB retrieval
    where_filter = (
        {"document_id": document_id} if document_id else None
    )

    chroma_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k * 2, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    if not chroma_results["documents"][0]:
        return {
            "answer": "No relevant medical information found "
                      "for your question.",
            "sources": [],
            "critical_alerts": [],
            "disclaimer": "",
        }

    # Step 3: Prepare chunks and embeddings for hybrid retrieval
    raw_chunks = []
    chunk_embeddings_list = []

    for j, (doc, meta, emb, dist) in enumerate(zip(
        chroma_results["documents"][0],
        chroma_results["metadatas"][0],
        chroma_results["embeddings"][0],
        chroma_results["distances"][0],
    )):
        raw_chunks.append({
            "text": doc,
            "section": meta.get("section", "unknown"),
            "chunk_type": meta.get("chunk_type", "unknown"),
            "document_id": meta.get("document_id"),
            "filename": meta.get("filename", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
        })
        chunk_embeddings_list.append(emb)

    chunk_embeddings = np.array(chunk_embeddings_list)

    # Step 4: Hybrid retrieval
    retrieved_chunks = hybrid_medical_retrieval(
        query=question,
        query_embedding=query_embedding,
        chunks=raw_chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )

    # Step 5: Load clinical entities (Stage 2)
    clinical_entities = None
    if document_id:
        try:
            clinical_entities = get_clinical_entities(document_id)
        except Exception as e:
            logger.warning(f"Could not load clinical entities: {e}")

    # Step 6: Load KB context (Stage 3)
    kb_context = None
    if include_kb and document_id:
        try:
            kb_context = enrich_with_knowledge_base(document_id)
        except Exception as e:
            logger.warning(f"KB context unavailable: {e}")

    # Step 7: Generate safe medical answer
    result = generate_medical_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        clinical_entities=clinical_entities,
        kb_context=kb_context,
    )

    result["question"] = question
    result["document_id"] = document_id

    logger.info(
        f"Medical RAG complete | "
        f"answer_chars={len(result['answer'])} | "
        f"sources={len(result['sources'])} | "
        f"alerts={len(result['critical_alerts'])}"
    )

    return result