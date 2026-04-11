import json
from datetime import datetime
from backend.database.db import SessionLocal, MedicalDocument
from backend.logger import get_logger

logger = get_logger("storage.document_store")


def save_medical_document(
    filename: str,
    file_type: str,
    document_type: str,
    page_count: int | None,
    word_count: int,
    file_size_kb: float,
    extracted_text: str,
    medical_metadata: dict,
) -> int:
    """Saves medical document with all metadata to SQLite."""
    db = SessionLocal()
    try:
        record = MedicalDocument(
            filename=filename,
            file_type=file_type,
            document_type=document_type,
            page_count=page_count,
            word_count=word_count,
            file_size_kb=file_size_kb,
            upload_timestamp=datetime.utcnow(),
            extracted_text=extracted_text,
            has_pii=medical_metadata.get("has_pii", False),
            urgency_level=medical_metadata.get("urgency_level", "routine"),
            medical_facility=medical_metadata.get("medical_facility"),
            document_date=medical_metadata.get("document_date"),
            medical_metadata=json.dumps(medical_metadata),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(
            f"Document saved | id={record.id} | "
            f"type={document_type} | urgency={record.urgency_level}"
        )
        return record.id
    except Exception as e:
        db.rollback()
        logger.error(f"Save failed: {e}", exc_info=True)
        raise
    finally:
        db.close()


def get_all_documents() -> list[dict]:
    db = SessionLocal()
    try:
        records = db.query(MedicalDocument).all()
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "file_type": r.file_type,
                "document_type": r.document_type,
                "page_count": r.page_count,
                "word_count": r.word_count,
                "file_size_kb": r.file_size_kb,
                "urgency_level": r.urgency_level,
                "has_pii": r.has_pii,
                "medical_facility": r.medical_facility,
                "upload_timestamp": str(r.upload_timestamp),
            }
            for r in records
        ]
    finally:
        db.close()


def get_document_by_id(doc_id: int) -> MedicalDocument | None:
    db = SessionLocal()
    try:
        return db.query(MedicalDocument).filter(
            MedicalDocument.id == doc_id
        ).first()
    finally:
        db.close()