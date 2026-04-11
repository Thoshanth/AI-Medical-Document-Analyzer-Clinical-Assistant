import re
from backend.logger import get_logger

logger = get_logger("medical_rag.chunker")

# Medical section headers — these define natural chunk boundaries
SECTION_HEADERS = [
    r"^(chief complaint|cc)[\s:]+",
    r"^(history of present illness|hpi)[\s:]+",
    r"^(past medical history|pmh)[\s:]+",
    r"^(medications?|current medications?)[\s:]+",
    r"^(allergies?)[\s:]+",
    r"^(physical examination|pe)[\s:]+",
    r"^(vital signs?|vitals?)[\s:]+",
    r"^(laboratory results?|lab results?|labs?)[\s:]+",
    r"^(assessment|impression)[\s:]+",
    r"^(plan|treatment plan)[\s:]+",
    r"^(diagnosis|diagnoses)[\s:]+",
    r"^(subjective)[\s:]+",
    r"^(objective)[\s:]+",
    r"^(discharge (summary|instructions?))[\s:]+",
    r"^(follow.?up)[\s:]+",
]

SECTION_PATTERN = re.compile(
    "|".join(SECTION_HEADERS),
    re.IGNORECASE | re.MULTILINE
)

# Lab value pattern — keeps value + reference + interpretation together
LAB_PATTERN = re.compile(
    r"([A-Za-z\s/]+):\s*([\d.]+)\s*([a-zA-Z/%]+)?\s*"
    r"(?:\(?(normal|reference|ref)[\s:]*[\d.\-\s]+\)?)?",
    re.IGNORECASE
)


def chunk_by_medical_sections(text: str) -> list[dict]:
    """
    Splits medical document by clinical sections.

    Each section becomes one chunk with its header as metadata.
    This preserves clinical context — lab values stay with their
    section, medications stay together, etc.

    Returns list of dicts with text and section metadata.
    """
    logger.debug("Chunking by medical sections")
    chunks = []

    # Find all section boundaries
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        logger.debug("No medical sections found — using paragraph chunking")
        return chunk_by_paragraphs(text)

    # Extract each section
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        section_name = match.group(0).strip().rstrip(":")

        if len(section_text) > 20:
            chunks.append({
                "text": section_text,
                "section": section_name,
                "chunk_type": "medical_section",
                "chunk_index": i,
            })

    # Handle text before first section
    if matches and matches[0].start() > 50:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.insert(0, {
                "text": preamble,
                "section": "document_header",
                "chunk_type": "preamble",
                "chunk_index": -1,
            })

    logger.info(
        f"Section chunking | sections={len(chunks)}"
    )
    return chunks


def chunk_by_paragraphs(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """
    Fallback chunker for documents without clear sections.
    Uses paragraph boundaries with word overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks = []
    current_words = []
    chunk_index = 0

    for para in paragraphs:
        para_words = para.split()

        if len(current_words) + len(para_words) <= chunk_size:
            current_words.extend(para_words)
        else:
            if current_words:
                chunks.append({
                    "text": " ".join(current_words),
                    "section": "paragraph",
                    "chunk_type": "paragraph",
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                current_words = current_words[-overlap:] + para_words
            else:
                current_words = para_words

    if current_words:
        chunks.append({
            "text": " ".join(current_words),
            "section": "paragraph",
            "chunk_type": "paragraph",
            "chunk_index": chunk_index,
        })

    logger.info(f"Paragraph chunking | chunks={len(chunks)}")
    return chunks


def chunk_medical_document(
    text: str,
    doc_type: str,
) -> list[dict]:
    """
    Main entry point — selects chunking strategy based on document type.

    clinical_note → section-based chunking (SOAP sections)
    lab_report    → section-based chunking (preserves lab values)
    prescription  → section-based chunking (preserves drug lists)
    research_paper → paragraph chunking (no standard sections)
    fhir_record   → section-based chunking
    general_medical → paragraph chunking
    """
    logger.info(
        f"Medical chunking | doc_type={doc_type} | chars={len(text)}"
    )

    section_types = [
        "clinical_note", "lab_report",
        "prescription", "fhir_record"
    ]

    if doc_type in section_types:
        chunks = chunk_by_medical_sections(text)
        # If section chunking found nothing, fall back to paragraphs
        if not chunks:
            chunks = chunk_by_paragraphs(text)
    else:
        chunks = chunk_by_paragraphs(text)

    # Filter out very short chunks
    chunks = [c for c in chunks if len(c["text"].split()) >= 10]

    logger.info(
        f"Chunking complete | "
        f"strategy={'section' if doc_type in section_types else 'paragraph'} | "
        f"chunks={len(chunks)}"
    )
    return chunks