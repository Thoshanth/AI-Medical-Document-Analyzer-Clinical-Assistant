import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  generateSoapNote, generateDifferential,
  generateMedReport, generateLabReport,
  generateFullReport, getSavedReport, getDocuments,
} from "../api";
import DocSelector from "../components/DocSelector";
import LoadingSpinner from "../components/LoadingSpinner";

const REPORT_TYPES = [
  { id: "soap", label: "SOAP Note", fn: generateSoapNote, desc: "Subjective, Objective, Assessment, Plan" },
  { id: "differential", label: "Differential Diagnosis", fn: generateDifferential, desc: "Ranked possible diagnoses" },
  { id: "medication", label: "Medication Review", fn: generateMedReport, desc: "Pharmacist-style drug review" },
  { id: "lab", label: "Lab Interpretation", fn: generateLabReport, desc: "Lab values with clinical context" },
];

export default function ReportsPage({ selectedDoc: propDoc, showToast }) {
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(propDoc);
  const [activeReport, setActiveReport] = useState(null);
  const [reportContent, setReportContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingType, setLoadingType] = useState("");

  useEffect(() => {
    getDocuments().then(setDocuments).catch(() => {});
  }, []);

  useEffect(() => { setActiveDoc(propDoc); }, [propDoc]);

  const generateReport = async (type) => {
    if (!activeDoc) {
      showToast("Please select a document first", "error");
      return;
    }
    setLoading(true);
    setLoadingType(type.label);
    setActiveReport(type.id);
    try {
      const result = await type.fn(activeDoc.id);
      setReportContent(result);
      showToast(`✅ ${type.label} generated`, "success");
    } catch (e) {
      showToast(`Failed to generate ${type.label}`, "error");
      setReportContent(null);
    } finally {
      setLoading(false);
      setLoadingType("");
    }
  };

  const generateAll = async () => {
    if (!activeDoc) { showToast("Select a document first", "error"); return; }
    setLoading(true);
    setLoadingType("Full Report (all 4 types)");
    setActiveReport("full");
    try {
      const result = await generateFullReport(activeDoc.id);
      setReportContent(result);
      showToast("✅ Full report generated", "success");
    } catch {
      showToast("Failed to generate full report", "error");
    } finally {
      setLoading(false);
      setLoadingType("");
    }
  };

  const getReportText = () => {
    if (!reportContent) return null;
    if (activeReport === "full") {
      return Object.entries(reportContent.reports || {})
        .map(([type, data]) => `# ${type.replace(/_/g, " ").toUpperCase()}\n\n${data.report || ""}`)
        .join("\n\n---\n\n");
    }
    return reportContent.report || JSON.stringify(reportContent, null, 2);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Clinical Reports</h1>
        <DocSelector documents={documents} selectedDoc={activeDoc} onSelect={setActiveDoc} />
      </div>

      {/* Report type buttons */}
      <div className="report-grid">
        {REPORT_TYPES.map((type) => (
          <button
            key={type.id}
            className={`report-card ${activeReport === type.id ? "active" : ""}`}
            onClick={() => generateReport(type)}
            disabled={loading}
          >
            <h3>{type.label}</h3>
            <p>{type.desc}</p>
          </button>
        ))}

        <button
          className={`report-card full-report ${activeReport === "full" ? "active" : ""}`}
          onClick={generateAll}
          disabled={loading}
        >
          <h3>🗂 Full Report</h3>
          <p>Generate all 4 reports at once</p>
        </button>
      </div>

      {/* Report content */}
      {loading && <LoadingSpinner text={`Generating ${loadingType}...`} />}

      {reportContent && !loading && (
        <div className="report-output">
          <div className="report-output-header">
            <h2>
              {activeReport === "full" ? "Full Clinical Report" :
                REPORT_TYPES.find(t => t.id === activeReport)?.label}
            </h2>
            {reportContent.disclaimer && (
              <div className="disclaimer-box">
                {reportContent.disclaimer}
              </div>
            )}
          </div>
          <div className="report-body">
            <ReactMarkdown>{getReportText()}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}