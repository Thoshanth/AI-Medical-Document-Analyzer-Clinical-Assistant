from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Text, Boolean, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./medical_platform.db"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class MedicalDocument(Base):
    __tablename__ = "medical_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_type = Column(String)
    document_type = Column(String)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer)
    file_size_kb = Column(Float)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    has_pii = Column(Boolean, default=False)
    urgency_level = Column(String, default="routine")
    medical_facility = Column(String, nullable=True)
    document_date = Column(String, nullable=True)
    extracted_text = Column(Text)
    medical_metadata = Column(Text, nullable=True)


class ClinicalEntities(Base):
    """
    Stores structured clinical entities extracted by Stage 2 NLP.
    Linked to MedicalDocument by document_id.
    Each entity type stored as JSON string for flexibility.
    """
    __tablename__ = "clinical_entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("medical_documents.id"))
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # Six clinical entity types — stored as JSON arrays
    symptoms = Column(Text, default="[]")
    diagnoses = Column(Text, default="[]")
    medications = Column(Text, default="[]")
    lab_values = Column(Text, default="[]")
    vitals = Column(Text, default="[]")
    procedures = Column(Text, default="[]")

    # ICD-10 mappings — stored as JSON object
    icd10_codes = Column(Text, default="{}")

    # Summary stats
    total_entities_found = Column(Integer, default=0)
    clinical_complexity = Column(String, default="low")  # low/medium/high


def init_db():
    Base.metadata.create_all(bind=engine)