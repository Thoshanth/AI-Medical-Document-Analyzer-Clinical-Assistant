import numpy as np
import time
from backend.logger import get_logger

logger = get_logger("medical_rag.embedder")

# We try PubMedBERT first (medical domain)
# Fall back to MiniLM if not available
PUBMED_MODEL = "NLP4Science/pubmedbert-small-finetuned-pubmed"
FALLBACK_MODEL = "all-MiniLM-L6-v2"

_embedder = None


def get_embedder():
    """
    Lazy loads the embedding model.
    Tries PubMedBERT first, falls back to MiniLM.
    Cached after first load for performance.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    from sentence_transformers import SentenceTransformer

    try:
        logger.info(f"Loading PubMedBERT embedder | model={PUBMED_MODEL}")
        start = time.time()
        _embedder = SentenceTransformer(PUBMED_MODEL)
        elapsed = round(time.time() - start, 2)
        logger.info(f"PubMedBERT loaded | time={elapsed}s")
    except Exception as e:
        logger.warning(
            f"PubMedBERT load failed: {e} | "
            f"falling back to MiniLM"
        )
        _embedder = SentenceTransformer(FALLBACK_MODEL)
        logger.info("MiniLM fallback loaded")

    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of texts using medical domain model.
    Returns 2D numpy array of shape (n_texts, embedding_dim).
    """
    embedder = get_embedder()
    logger.debug(f"Embedding {len(texts)} medical texts")
    start = time.time()

    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )

    elapsed = round(time.time() - start, 3)
    logger.info(
        f"Embedding complete | "
        f"texts={len(texts)} | "
        f"dim={vectors.shape[1]} | "
        f"time={elapsed}s"
    )
    return vectors


def embed_query(query: str) -> np.ndarray:
    """Embeds a single query text."""
    return embed_texts([query])[0]