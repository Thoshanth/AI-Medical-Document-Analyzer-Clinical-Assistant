import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from backend.logger import get_logger

logger = get_logger("medical_rag.retriever")

# Medical terms that boost retrieval score when matched exactly
MEDICAL_BOOST_TERMS = [
    "diagnosis", "medication", "drug", "symptom",
    "lab", "result", "value", "normal", "abnormal",
    "treatment", "procedure", "allergy", "dosage",
]


def hybrid_medical_retrieval(
    query: str,
    query_embedding: np.ndarray,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
    doc_type_filter: str = None,
) -> list[dict]:
    """
    Three-signal hybrid retrieval for medical documents.

    Signal 1 — Semantic similarity (PubMedBERT vectors)
    Signal 2 — BM25 keyword matching
    Signal 3 — Medical entity boost (exact medical term matches)

    Combines all three using weighted scoring.
    """
    if not chunks:
        return []

    logger.info(
        f"Hybrid retrieval | query='{query[:50]}' | "
        f"chunks={len(chunks)} | top_k={top_k}"
    )

    chunk_texts = [c["text"] for c in chunks]

    # ── Signal 1: Semantic similarity ────────────────────────────
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        chunk_embeddings
    )[0]

    # ── Signal 2: BM25 keyword matching ──────────────────────────
    tokenized_chunks = [t.lower().split() for t in chunk_texts]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to 0-1
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_normalized = bm25_scores / bm25_max
    else:
        bm25_normalized = bm25_scores

    # ── Signal 3: Medical entity boost ───────────────────────────
    query_lower = query.lower()
    medical_boost = np.zeros(len(chunks))

    for i, chunk in enumerate(chunks):
        chunk_lower = chunk["text"].lower()
        boost = 0

        # Boost for medical term matches
        for term in MEDICAL_BOOST_TERMS:
            if term in query_lower and term in chunk_lower:
                boost += 0.1

        # Boost for section type relevance
        section = chunk.get("section", "").lower()
        if "lab" in query_lower and "lab" in section:
            boost += 0.2
        if "medication" in query_lower and "medication" in section:
            boost += 0.2
        if "diagnosis" in query_lower and "assessment" in section:
            boost += 0.2
        if "vital" in query_lower and "vital" in section:
            boost += 0.2

        medical_boost[i] = min(boost, 0.5)  # cap boost

    # ── Combine signals ───────────────────────────────────────────
    # Weights: semantic 50%, BM25 35%, medical boost 15%
    combined_scores = (
        0.50 * similarities +
        0.35 * bm25_normalized +
        0.15 * medical_boost
    )

    # Get top_k indices
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if combined_scores[idx] > 0:
            result = {
                **chunks[idx],
                "retrieval_score": round(float(combined_scores[idx]), 4),
                "semantic_score": round(float(similarities[idx]), 4),
                "bm25_score": round(float(bm25_normalized[idx]), 4),
                "medical_boost": round(float(medical_boost[idx]), 4),
            }
            results.append(result)

    logger.info(
        f"Retrieval complete | "
        f"results={len(results)} | "
        f"top_score={results[0]['retrieval_score'] if results else 0}"
    )
    return results