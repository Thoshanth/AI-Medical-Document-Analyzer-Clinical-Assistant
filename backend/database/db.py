from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Text, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./medical_platform.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class MedicalDocument(Base):
    """
    Stores every uploaded medical document with rich metadata.
    More fields than Project 1 to capture medical context.
    """
    __tablename__ = "medical_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_type = Column(String)           # pdf, txt, json
    document_type = Column(String)       # lab_report, prescription, etc.
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer)
    file_size_kb = Column(Float)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)

    # Medical specific fields
    has_pii = Column(Boolean, default=False)    # contains patient info?
    urgency_level = Column(String, default="routine")  # emergency/urgent/routine
    medical_facility = Column(String, nullable=True)   # hospital/clinic name
    document_date = Column(String, nullable=True)      # date on document

    # Full extracted text
    extracted_text = Column(Text)

    # Raw medical metadata as JSON string
    medical_metadata = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)