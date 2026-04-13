
# 🏥 AI Medical Document Analyzer & Clinical Assistant

A production-grade medical AI platform built stage by stage — from raw 
medical document ingestion to a 5-agent clinical system with safety 
guardrails, drug interaction checking, knowledge graphs, and RAGAS 
evaluation. The most complex AI engineering project in this portfolio.

---

## ⚠️ Medical Disclaimer

This platform is built for **educational and research purposes only**.
It is a clinical decision **support tool** — not a replacement for 
licensed medical professionals. All outputs must be reviewed by 
qualified healthcare providers before any clinical use.

---

## 🎯 What This Project Does

Upload any medical document — patient reports, lab results, 
prescriptions, FHIR records, clinical notes, research papers — and get:

- Structured clinical entity extraction (symptoms, diagnoses, 
  medications, lab values)
- Drug interaction safety checking with severity classification
- ICD-10 code mapping for all extracted diagnoses
- Evidence-based medical RAG with PubMedBERT embeddings
- Medical knowledge graph with clinical relationship traversal
- AI-generated SOAP notes, differential diagnoses, medication reviews
- 5-agent clinical system (Triage → Diagnosis → Pharmacist → 
  Research → Safety)
- RAGAS evaluation with medical-specific safety metrics
- Emergency detection with immediate crisis resource escalation

---

## 🏗️ Architecture Overview
User Request
│
▼
Stage 9: Medical Safety Guardrails (6 layers)
│
▼
FastAPI Backend (12 integrated stages)
│
├── Stage 1:  Medical Document Ingestion
├── Stage 2:  Clinical NLP Pipeline
├── Stage 3:  Medical Knowledge Base
├── Stage 4:  Medical RAG (PubMedBERT)
├── Stage 5:  Drug Interaction Checker
├── Stage 6:  Medical Knowledge Graph
├── Stage 7:  Clinical Report Generator
├── Stage 8:  5-Agent Clinical System
├── Stage 9:  Medical Safety Guardrails
├── Stage 10: Fine-Tuning Pipeline
├── Stage 11: RAGAS Clinical Evaluation
└── Stage 12: React Full Stack Dashboard

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11 |
| **LLM** | NVIDIA Nemotron 120B via OpenRouter (free) |
| **Medical Embeddings** | PubMedBERT / sentence-transformers |
| **Vector DB** | ChromaDB (persistent) |
| **RAG Framework** | LlamaIndex |
| **Agent Framework** | LangGraph |
| **Knowledge Graph** | NetworkX |
| **Drug Data** | OpenFDA API + curated database |
| **ICD-10** | simple-icd-10 library |
| **FHIR Parsing** | fhir.resources |
| **Fine-tuning** | Unsloth + QLoRA + HuggingFace |
| **Evaluation** | RAGAS framework |
| **Database** | SQLite + SQLAlchemy |
| **Frontend** | React + Lucide Icons |
| **Deployment** | Docker + Docker Compose |

---

## 🚀 Stages Built

### Stage 1 — Medical Document Ingestion
- Smart PDF extraction with PyMuPDF + pdfplumber fallback
- FHIR JSON parser (Patient, Condition, Medication, Observation, Bundle)
- Six document type classification: lab_report, prescription,
  clinical_note, fhir_record, research_paper, general_medical
- Medical abbreviation expansion (30+ clinical abbreviations)
- Urgency detection: emergency / urgent / routine
- Medical PII detection: patient name, DOB, MRN, insurance ID, Aadhaar
- Emergency alert in API response for critical documents

### Stage 2 — Clinical NLP Pipeline
- MiniMax-powered extraction of 6 entity types:
  symptoms, diagnoses, medications, lab values, vitals, procedures
- ICD-10 code mapping with 3-tier lookup (fast cache / library / LLM)
- Medication normalization: frequency standardization and drug class
- Lab value interpretation with normal ranges and critical value detection
- Clinical complexity assessment: low / medium / high
- Clinical alerts for critical labs, high-risk medications, severe symptoms

### Stage 3 — Medical Knowledge Base
- Curated disease database (6 conditions with full clinical profiles)
- Drug database with 8 medications (local + OpenFDA API + disk cache)
- Three-tier drug lookup: local → cache → OpenFDA live API
- Symptom checker with severity scoring and triage level assessment
- Clinical summary with treatment gaps and monitoring requirements

### Stage 4 — Medical RAG Pipeline
- Medical-aware chunking preserving lab sections and SOAP note structure
- PubMedBERT embeddings (fallback to MiniLM)
- Three-signal hybrid retrieval: semantic + BM25 + medical entity boost
- Safe answer generation with mandatory medical disclaimer
- Critical finding detection and alerting in every response
- Full context injection: Stage 2 entities + Stage 3 KB enrichment

### Stage 5 — Drug Interaction Checker
- Curated interaction database with 15 critical drug pairs
- OpenFDA live drug label API integration
- LLM fallback with explicit AI disclaimer for unknown combinations
- Checks all medication pairs from Stage 2 clinical entities
- Severity sorting: major interactions flagged first
- Clinical recommendations: URGENT / CAUTION / MINOR / CLEAR
- Standalone pair checker endpoint for quick lookups

### Stage 6 — Medical Knowledge Graph
- Pre-built foundation graph: 54 nodes, 68 edges of medical knowledge
- 7 medical node types: symptom, disease, drug, lab_test, procedure,
  anatomy, finding
- 14 clinical relationship types: INDICATES, TREATED_BY,
  COMPLICATION_OF, MONITORED_BY, CONTRAINDICATED_IN, and more
- LLM + Stage 2 entity extraction for document-specific graph layer
- BFS traversal: complications, treatment pathways, differential diagnosis
- Patient clinical picture: complications + monitoring + contraindications
- Combined foundation + document graph for multi-hop reasoning

### Stage 7 — Clinical Report Generator
- SOAP note: S/O/A/P sections from Stage 2 entities + Stage 3 KB
- Differential diagnosis report: ranked diagnoses with Stage 6 graph
- Medication review report: pharmacist-style with Stage 5 interactions
- Lab interpretation report: critical value flagging + pattern analysis
- Full report endpoint generates all 4 reports in one call
- Reports saved to disk for retrieval without LLM re-call
- Mandatory medical disclaimer on every report

### Stage 8 — 5-Agent Clinical System (LangGraph)
- Triage Agent: urgency assessment, emergency detection, fast-track
- Diagnosis Agent: differential diagnosis with Stage 6 graph traversal
- Pharmacist Agent: medication safety with Stage 5 interactions
- Research Agent: evidence-based findings with Stage 4 Medical RAG
- Safety Agent: quality control, ethics review, final approval
- Emergency shortcut: triage → safety (skips non-urgent agents)
- Revision loop: safety agent can send back to any agent (max 3 iterations)
- show_agent_trace reveals full reasoning of each agent

### Stage 9 — Medical Safety Guardrails (6 Layers)
- Layer 1: Input validation (dangerous self-treatment, misinformation,
  prompt injection)
- Layer 2: Emergency detection with immediate crisis resources
  (cardiac, stroke, overdose, suicide)
- Layer 3: Medical PII shield (patient name, DOB, MRN, insurance ID,
  Aadhaar, PAN card)
- Layer 4: Medical scope enforcement (keyword fast-path + LLM judge)
- Layer 5: Hallucination detection (suspicious precision, fabricated
  citations, overconfidence)
- Layer 6: Medical PII redaction from LLM outputs + disclaimer injection
- Applied to: /medical-rag/query and /clinical-agents/analyze

### Stage 10 — Fine-Tuning Pipeline
- Medical QA dataset generator (templates + KB + document-specific)
- 10 high-quality curated template QA pairs across clinical domains
- KB-derived QA pairs from all diseases and drugs in Stage 3 database
- Document-specific QA generation via LLM from uploaded files
- QLoRA fine-tuning on BioMistral-7B (simulation mode by default)
- Actual fine-tuning support with Unsloth for GPU environments
- Alpaca instruction format for all training examples
- Training status tracking with disk persistence

### Stage 11 — RAGAS Clinical Evaluation
- Golden evaluation dataset: 8 static + document-specific QA pairs
- RAGAS metrics: faithfulness, answer relevancy, context precision,
  context recall
- Emergency detection evaluation: sensitivity + specificity measurement
- Disclaimer compliance: 100% required for medical AI
- Clinical accuracy: LLM-as-judge vs ground truth answers
- Medication safety language evaluation
- Overall grade: A/B/C/D with production readiness assessment
- Actionable recommendations for improvement

### Stage 12 — Full Stack React Dashboard
- Dark professional medical dashboard theme
- Sidebar navigation: Documents, Chat, Reports, Drug Safety, Agents,
  Evaluation
- DocumentsPage: upload + one-click full pipeline runner
- ChatPage: medical RAG chat with citations + safety warnings
- ReportsPage: all 4 report types + full combined report
- DrugCheckPage: document check + specific pair checker
- AgentsPage: 5-agent system with collapsible reasoning trace
- EvaluationPage: RAGAS metric bars + emergency test + grade display

---

## 📡 API Endpoints

| Method | Endpoint | Stage | Description |
|---|---|---|---|
| POST | `/upload` | 1 | Upload medical document |
| GET | `/documents` | 1 | List all documents |
| GET | `/documents/{id}` | 1 | Get document details |
| POST | `/analyze/{id}` | 2 | Run clinical NLP |
| GET | `/analyze/{id}` | 2 | Get stored entities |
| POST | `/knowledge/enrich/{id}` | 3 | KB enrichment |
| GET | `/knowledge/drug/{name}` | 3 | Drug lookup |
| GET | `/knowledge/disease/{name}` | 3 | Disease lookup |
| POST | `/medical-rag/index/{id}` | 4 | Index for RAG |
| POST | `/medical-rag/query` | 4 | Medical RAG query |
| POST | `/interactions/check/{id}` | 5 | Check all medications |
| POST | `/interactions/check-pair` | 5 | Check drug pair |
| POST | `/medical-graph/build/{id}` | 6 | Build graph |
| POST | `/medical-graph/query` | 6 | Graph query |
| GET | `/medical-graph/patient-summary/{id}` | 6 | Patient summary |
| POST | `/reports/soap/{id}` | 7 | SOAP note |
| POST | `/reports/differential/{id}` | 7 | Differential diagnosis |
| POST | `/reports/medication/{id}` | 7 | Medication review |
| POST | `/reports/lab/{id}` | 7 | Lab interpretation |
| POST | `/reports/full/{id}` | 7 | All 4 reports |
| POST | `/clinical-agents/analyze` | 8 | 5-agent analysis |
| POST | `/safety/check-input` | 9 | Input safety check |
| POST | `/safety/emergency-check` | 9 | Emergency detection |
| POST | `/fine-tuning/generate-dataset` | 10 | Build training data |
| POST | `/fine-tuning/train` | 10 | Run fine-tuning |
| POST | `/fine-tuning/evaluate` | 10 | Evaluate model |
| POST | `/evaluation/run/{id}` | 11 | Full RAGAS evaluation |
| POST | `/evaluation/emergency-test` | 11 | Emergency detection test |
| GET | `/evaluation/results/{id}` | 11 | Get saved results |
| GET | `/health` | — | Health check |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-medical-platform.git
cd ai-medical-platform
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a free OpenRouter API key at: https://openrouter.ai/keys

### 5. Run the backend
```bash
uvicorn backend.main:app --reload
```

API available at: `http://127.0.0.1:8000`
Interactive docs at: `http://127.0.0.1:8000/docs`

### 6. Run the frontend
```bash
cd frontend
npm install
npm start
```

Dashboard at: `http://localhost:3000`

### 7. Run with Docker
```bash
docker compose up --build
```

---

## 🧪 Quick Test Flow

```bash
# 1. Upload a medical document
curl -X POST http://localhost:8000/upload -F "file=@report.pdf"
# Note the document_id

# 2. Run clinical NLP
curl -X POST http://localhost:8000/analyze/1

# 3. Run knowledge base enrichment
curl -X POST http://localhost:8000/knowledge/enrich/1

# 4. Index for medical RAG
curl -X POST http://localhost:8000/medical-rag/index/1

# 5. Check drug interactions
curl -X POST http://localhost:8000/interactions/check/1

# 6. Build medical knowledge graph
curl -X POST http://localhost:8000/medical-graph/build/1

# 7. Generate full clinical report
curl -X POST http://localhost:8000/reports/full/1

# 8. Run 5-agent clinical analysis
curl -X POST "http://localhost:8000/clinical-agents/analyze?document_id=1&question=What+is+the+clinical+assessment?&show_agent_trace=true"

# 9. Run RAGAS evaluation
curl -X POST http://localhost:8000/evaluation/run/1
```

---

## 📁 Project Structure
ai-medical-platform/
├── backend/
│   ├── main.py                        # FastAPI app + all endpoints
│   ├── logger.py                      # Centralized structured logging
│   ├── llm_client.py                  # LLM client with fallback chain
│   ├── database/
│   │   └── db.py                      # SQLite + SQLAlchemy schema
│   ├── ingestion/
│   │   ├── extractor.py               # Medical document extraction
│   │   ├── fhir_parser.py             # FHIR JSON parser
│   │   ├── classifier.py              # Document type classification
│   │   ├── medical_metadata.py        # Urgency + PII detection
│   │   └── cleaner.py                 # Medical text normalization
│   ├── clinical_nlp/
│   │   ├── entity_extractor.py        # 6-type clinical entity extraction
│   │   ├── icd_mapper.py              # ICD-10 code mapping
│   │   ├── medication_parser.py       # Drug normalization
│   │   ├── lab_interpreter.py         # Lab value interpretation
│   │   └── nlp_pipeline.py            # Full NLP orchestration
│   ├── knowledge_base/
│   │   ├── disease_db.py              # Curated disease database
│   │   ├── drug_db.py                 # Drug DB + OpenFDA API
│   │   ├── symptom_checker.py         # Symptom → condition mapping
│   │   └── kb_pipeline.py             # KB enrichment pipeline
│   ├── medical_rag/
│   │   ├── medical_chunker.py         # Medical-aware chunking
│   │   ├── medical_embedder.py        # PubMedBERT embeddings
│   │   ├── medical_retriever.py       # Hybrid 3-signal retrieval
│   │   ├── answer_generator.py        # Safe medical answer generation
│   │   └── rag_pipeline.py            # Full RAG orchestration
│   ├── drug_interaction/
│   │   ├── interaction_db.py          # 15 curated interaction pairs
│   │   ├── fda_checker.py             # OpenFDA interaction API
│   │   ├── llm_checker.py             # LLM fallback checker
│   │   └── interaction_pipeline.py    # All-medication check
│   ├── medical_graph/
│   │   ├── graph_schema.py            # Node + edge type definitions
│   │   ├── foundation_graph.py        # Pre-built medical knowledge graph
│   │   ├── graph_extractor.py         # LLM entity extraction
│   │   ├── graph_store.py             # NetworkX persistence
│   │   ├── graph_traversal.py         # Clinical reasoning traversal
│   │   └── graph_pipeline.py          # Graph build + query
│   ├── report_generator/
│   │   ├── soap_generator.py          # SOAP note generation
│   │   ├── differential_generator.py  # Differential diagnosis report
│   │   ├── medication_report.py       # Pharmacist medication review
│   │   ├── lab_report_generator.py    # Lab interpretation report
│   │   └── report_pipeline.py         # Full report orchestration
│   ├── clinical_agents/
│   │   ├── clinical_state.py          # LangGraph shared state
│   │   ├── triage_agent.py            # Triage Agent
│   │   ├── diagnosis_agent.py         # Diagnosis Agent
│   │   ├── pharmacist_agent.py        # Pharmacist Agent
│   │   ├── research_agent.py          # Research Agent
│   │   ├── safety_agent.py            # Safety & Ethics Agent
│   │   ├── clinical_graph.py          # LangGraph workflow
│   │   └── agent_pipeline.py          # Entry point
│   ├── medical_safety/
│   │   ├── input_validator.py         # Layer 1: input validation
│   │   ├── emergency_detector.py      # Layer 2: emergency detection
│   │   ├── medical_pii_shield.py      # Layer 3: medical PII
│   │   ├── hallucination_checker.py   # Layer 4: hallucination check
│   │   ├── disclaimer_injector.py     # Layer 5: disclaimer injection
│   │   ├── scope_enforcer.py          # Layer 6: scope enforcement
│   │   └── safety_pipeline.py         # Safety orchestration
│   ├── fine_tuning/
│   │   ├── dataset_generator.py       # Medical QA dataset builder
│   │   ├── trainer.py                 # QLoRA fine-tuning
│   │   ├── evaluator.py               # Model evaluation
│   │   └── ft_pipeline.py             # Full pipeline
│   └── evaluation/
│       ├── golden_dataset.py          # Golden QA dataset
│       ├── ragas_evaluator.py         # RAGAS metrics
│       ├── medical_metrics.py         # Medical-specific metrics
│       └── eval_pipeline.py           # Full evaluation
├── frontend/
│   └── src/
│       ├── api.js                     # Complete API service
│       ├── App.js                     # Main app + routing
│       ├── components/
│       │   ├── Sidebar.js             # Navigation sidebar
│       │   ├── DocSelector.js         # Document selector
│       │   └── LoadingSpinner.js      # Loading component
│       └── pages/
│           ├── DocumentsPage.js       # Upload + pipeline runner
│           ├── ChatPage.js            # Medical RAG chat
│           ├── ReportsPage.js         # Report generation
│           ├── DrugCheckPage.js       # Drug interaction checker
│           ├── AgentsPage.js          # 5-agent system UI
│           └── EvaluationPage.js      # RAGAS evaluation UI
├── uploads/                           # Uploaded documents
├── chroma_db/                         # ChromaDB vector storage
├── graph_data/                        # Medical knowledge graphs
├── knowledge_cache/                   # OpenFDA API cache
├── fine_tuning_data/                  # Training datasets + adapters
├── evaluation_results/                # RAGAS evaluation reports
├── reports/                           # Generated clinical reports
├── logs/                              # Daily structured logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                               # API keys (not committed)

---

## 🔐 Medical Safety Features

**Input Safety (6 Layers):**
- Dangerous self-treatment request detection and blocking
- Emergency symptom detection → immediate crisis resources
- Medical PII detection (patient name, DOB, MRN, insurance, Aadhaar)
- Medical scope enforcement — stays focused on medical topics
- Hallucination signal detection in outputs
- Mandatory medical disclaimer injection on every response

**Emergency Response Protocol:**
- Cardiac arrest, stroke, overdose, respiratory emergency, suicide crisis
- Immediate response with emergency numbers (112, 911, 999)
- Crisis helplines for mental health emergencies
- Bypasses all normal processing — speed is critical

**Drug Safety:**
- 15 curated major/moderate/minor interaction pairs
- High-risk medication flagging (Warfarin, Insulin, Methotrexate)
- OpenFDA authoritative data integration
- Severity-sorted interaction reports

---

## 📊 Evaluation Metrics

| Metric | Description | Threshold |
|---|---|---|
| Faithfulness | Anti-hallucination score | ≥ 0.70 |
| Answer Relevancy | Question addressed | ≥ 0.70 |
| Context Precision | Retrieved chunks relevant | ≥ 0.60 |
| Context Recall | All info retrieved | ≥ 0.60 |
| Emergency Sensitivity | Emergencies detected | ≥ 0.95 |
| Disclaimer Compliance | Disclaimers present | 100% |
| Clinical Accuracy | LLM-judged accuracy | ≥ 0.70 |

---

## 🌿 Git Branch Strategy
main (stable)
└── phase1 (integration branch)
├── stage-1/medical-ingestion
├── stage-2/clinical-nlp
├── stage-3/medical-knowledge-base
├── stage-4/medical-rag
├── stage-5/drug-interaction
├── stage-6/medical-graph
├── stage-7/clinical-reports
├── stage-8/clinical-agents
├── stage-9/medical-safety
├── stage-10/fine-tuning
├── stage-11/evaluation
└── stage-12/fullstack

---

## 🎓 Learning Outcomes

By building this project you gain hands-on experience with:

- Medical document parsing including FHIR standard
- Clinical NLP — entity extraction, ICD-10 mapping, lab interpretation
- Domain-specific RAG with medical embeddings
- Drug interaction checking with multi-source validation
- Medical knowledge graphs with clinical relationship traversal
- Structured clinical report generation (SOAP notes)
- Multi-agent clinical systems with LangGraph
- Medical AI safety — emergency detection, PII handling, disclaimers
- QLoRA fine-tuning on biomedical models
- RAGAS evaluation with domain-specific medical metrics
- Full stack development with React medical dashboard

---

## 🤝 Contributing

This is a learning and portfolio project. Feel free to fork, extend,
or use as a foundation for your own medical AI research.

---

## 📄 License

MIT License — free to use for learning and portfolio purposes.

---

## 👨‍💻 Author

**M.S.N. Thoshanth Reddy**
B.Tech — Hyderabad Institute of Technology and Management (HITAM)

Project : AI Medical Document Analyzer (12 stages)

- GitHub: [Thoshanth](https://github.com/Thoshanth)
- LinkedIn: [linkedin.com/in/snthoshanthreddymandapati](https://www.linkedin.com/in/s-n-thoshanth-reddy-mandapati/)
- Email: mthoshanthreddy@gmail.com