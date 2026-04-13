import { useState, useEffect, useRef } from "react";
import {
  uploadDocument, getDocuments, analyzeDocument,
  indexDocument, buildGraph, checkInteractions,
} from "../api";
import LoadingSpinner from "../components/LoadingSpinner";
import {
  Upload, FileText, AlertTriangle,
  CheckCircle, Clock, RefreshCw
} from "lucide-react";

const URGENCY_COLORS = {
  emergency: "#ef5350",
  urgent: "#ffa726",
  routine: "#66bb6a",
};

export default function DocumentsPage({
  documents, setDocuments, selectedDoc,
  setSelectedDoc, showToast, onNavigate,
}) {
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(null);
  const [processingStep, setProcessingStep] = useState("");
  const fileRef = useRef();

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch {
      showToast("Failed to load documents", "error");
    }
  };

  const handleUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      showToast(`✅ Uploaded: ${result.filename}`, "success");

      if (result.medical_info?.urgency_level === "emergency") {
        showToast(
          "🚨 EMERGENCY indicators detected in document!",
          "error"
        );
      }

      await loadDocuments();
      setSelectedDoc({
        id: result.document_id,
        filename: result.filename,
        document_type: result.document_type,
        urgency_level: result.medical_info?.urgency_level,
      });
    } catch {
      showToast("Upload failed", "error");
    } finally {
      setUploading(false);
      fileRef.current.value = "";
    }
  };

  const runFullPipeline = async (doc) => {
    setProcessing(doc.id);
    const steps = [
      { label: "Running Clinical NLP...", fn: () => analyzeDocument(doc.id) },
      { label: "Indexing for RAG...", fn: () => indexDocument(doc.id) },
      { label: "Building Medical Graph...", fn: () => buildGraph(doc.id) },
      { label: "Checking Drug Interactions...", fn: () => checkInteractions(doc.id) },
    ];

    for (const step of steps) {
      setProcessingStep(step.label);
      try {
        await step.fn();
      } catch (e) {
        showToast(`${step.label} failed — continuing`, "error");
      }
    }

    showToast("✅ Full pipeline complete!", "success");
    setProcessing(null);
    setProcessingStep("");
    await loadDocuments();
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Medical Documents</h1>
        <p>Upload and process medical documents through all AI stages</p>
      </div>

      {/* Upload Area */}
      <div
        className="upload-area"
        onClick={() => fileRef.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleUpload(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.csv,.json"
          style={{ display: "none" }}
          onChange={(e) => handleUpload(e.target.files[0])}
        />
        {uploading ? (
          <LoadingSpinner text="Uploading and processing..." />
        ) : (
          <>
            <Upload size={32} color="#4fc3f7" />
            <p>Drop medical document here or click to upload</p>
            <span>PDF, TXT, CSV, JSON (FHIR)</span>
          </>
        )}
      </div>

      {/* Documents List */}
      <div className="section-header">
        <h2>Uploaded Documents ({documents.length})</h2>
        <button className="icon-btn" onClick={loadDocuments}>
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="doc-grid">
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={`doc-card ${selectedDoc?.id === doc.id ? "selected" : ""}`}
            onClick={() => setSelectedDoc(doc)}
          >
            <div className="doc-card-header">
              <FileText size={18} color="#4fc3f7" />
              <span
                className="urgency-badge"
                style={{
                  background:
                    URGENCY_COLORS[doc.urgency_level] || "#555",
                }}
              >
                {doc.urgency_level || "routine"}
              </span>
            </div>

            <p className="doc-filename">{doc.filename}</p>
            <p className="doc-type">{doc.document_type}</p>

            <div className="doc-meta">
              <span>{doc.word_count} words</span>
              <span>{doc.file_size_kb} KB</span>
              {doc.has_pii && (
                <span className="pii-badge">
                  <AlertTriangle size={12} /> PII
                </span>
              )}
            </div>

            {processing === doc.id ? (
              <div className="processing-status">
                <div className="mini-spinner" />
                <span>{processingStep}</span>
              </div>
            ) : (
              <button
                className="pipeline-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  runFullPipeline(doc);
                }}
              >
                Run Full Pipeline
              </button>
            )}
          </div>
        ))}

        {documents.length === 0 && (
          <div className="empty-state">
            <FileText size={48} color="#444" />
            <p>No documents yet. Upload one above.</p>
          </div>
        )}
      </div>
    </div>
  );
}