import axios from "axios";

const BASE = "http://127.0.0.1:8000";
const api = axios.create({ baseURL: BASE });

// ── Stage 1: Documents ────────────────────────────────────────────
export const uploadDocument = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const getDocuments = async () => {
  const res = await api.get("/documents");
  return res.data;
};

export const getDocument = async (id) => {
  const res = await api.get(`/documents/${id}`);
  return res.data;
};

// ── Stage 2: Clinical NLP ─────────────────────────────────────────
export const analyzeDocument = async (id) => {
  const res = await api.post(`/analyze/${id}`);
  return res.data;
};

export const getAnalysis = async (id) => {
  const res = await api.get(`/analyze/${id}`);
  return res.data;
};

// ── Stage 3: Knowledge Base ───────────────────────────────────────
export const enrichDocument = async (id) => {
  const res = await api.post(`/knowledge/enrich/${id}`);
  return res.data;
};

export const lookupDrug = async (name) => {
  const res = await api.get(`/knowledge/drug/${name}`);
  return res.data;
};

// ── Stage 4: Medical RAG ──────────────────────────────────────────
export const indexDocument = async (id) => {
  const res = await api.post(`/medical-rag/index/${id}`);
  return res.data;
};

export const medicalQuery = async (question, documentId = null, topK = 5) => {
  const params = { question, top_k: topK };
  if (documentId) params.document_id = documentId;
  const res = await api.post("/medical-rag/query", null, { params });
  return res.data;
};

// ── Stage 5: Drug Interactions ────────────────────────────────────
export const checkInteractions = async (id) => {
  const res = await api.post(`/interactions/check/${id}`);
  return res.data;
};

export const checkDrugPair = async (drugA, drugB) => {
  const res = await api.post("/interactions/check-pair", null, {
    params: { drug_a: drugA, drug_b: drugB },
  });
  return res.data;
};

// ── Stage 6: Medical Graph ────────────────────────────────────────
export const buildGraph = async (id) => {
  const res = await api.post(`/medical-graph/build/${id}`);
  return res.data;
};

export const graphQuery = async (question, documentId) => {
  const res = await api.post("/medical-graph/query", null, {
    params: { question, document_id: documentId },
  });
  return res.data;
};

export const getPatientSummary = async (id) => {
  const res = await api.get(`/medical-graph/patient-summary/${id}`);
  return res.data;
};

// ── Stage 7: Clinical Reports ─────────────────────────────────────
export const generateFullReport = async (id) => {
  const res = await api.post(`/reports/full/${id}`);
  return res.data;
};

export const generateSoapNote = async (id) => {
  const res = await api.post(`/reports/soap/${id}`);
  return res.data;
};

export const generateDifferential = async (id) => {
  const res = await api.post(`/reports/differential/${id}`);
  return res.data;
};

export const generateMedReport = async (id) => {
  const res = await api.post(`/reports/medication/${id}`);
  return res.data;
};

export const generateLabReport = async (id) => {
  const res = await api.post(`/reports/lab/${id}`);
  return res.data;
};

export const getSavedReport = async (id) => {
  const res = await api.get(`/reports/${id}`);
  return res.data;
};

// ── Stage 8: Clinical Agents ──────────────────────────────────────
export const runClinicalAgents = async (
  documentId,
  question,
  showTrace = false
) => {
  const res = await api.post("/clinical-agents/analyze", null, {
    params: {
      document_id: documentId,
      question,
      show_agent_trace: showTrace,
      max_iterations: 2,
    },
  });
  return res.data;
};

// ── Stage 9: Safety ───────────────────────────────────────────────
export const checkInputSafety = async (question) => {
  const res = await api.post("/safety/check-input", null, {
    params: { question },
  });
  return res.data;
};

// ── Stage 11: Evaluation ──────────────────────────────────────────
export const runEvaluation = async (id) => {
  const res = await api.post(`/evaluation/run/${id}`);
  return res.data;
};

export const testEmergencyDetection = async () => {
  const res = await api.post("/evaluation/emergency-test");
  return res.data;
};

export const getEvaluationResults = async (id) => {
  const res = await api.get(`/evaluation/results/${id}`);
  return res.data;
};

// ── Health ────────────────────────────────────────────────────────
export const getHealth = async () => {
  const res = await api.get("/health");
  return res.data;
};  