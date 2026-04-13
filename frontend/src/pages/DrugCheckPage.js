import { useState, useEffect } from "react";
import { checkInteractions, checkDrugPair, getDocuments } from "../api";
import DocSelector from "../components/DocSelector";
import LoadingSpinner from "../components/LoadingSpinner";
import { AlertTriangle, CheckCircle, Info } from "lucide-react";

const SEVERITY_CONFIG = {
  major: { color: "#ef5350", icon: "🔴", label: "MAJOR" },
  moderate: { color: "#ffa726", icon: "🟡", label: "MODERATE" },
  minor: { color: "#66bb6a", icon: "🟢", label: "MINOR" },
  none: { color: "#78909c", icon: "✅", label: "NONE" },
};

export default function DrugCheckPage({ selectedDoc: propDoc, showToast }) {
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(propDoc);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [drugA, setDrugA] = useState("");
  const [drugB, setDrugB] = useState("");
  const [pairResult, setPairResult] = useState(null);
  const [tab, setTab] = useState("document");

  useEffect(() => {
    getDocuments().then(setDocuments).catch(() => {});
  }, []);

  useEffect(() => { setActiveDoc(propDoc); }, [propDoc]);

  const checkDocumentInteractions = async () => {
    if (!activeDoc) { showToast("Select a document first", "error"); return; }
    setLoading(true);
    try {
      const data = await checkInteractions(activeDoc.id);
      setResult(data);
    } catch {
      showToast("Drug interaction check failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const checkPair = async () => {
    if (!drugA || !drugB) { showToast("Enter both drug names", "error"); return; }
    setLoading(true);
    try {
      const data = await checkDrugPair(drugA, drugB);
      setPairResult(data);
    } catch {
      showToast("Pair check failed", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Drug Safety Checker</h1>
        <p>Check drug interactions from documents or specific pairs</p>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button className={tab === "document" ? "tab active" : "tab"} onClick={() => setTab("document")}>
          Document Check
        </button>
        <button className={tab === "pair" ? "tab active" : "tab"} onClick={() => setTab("pair")}>
          Pair Check
        </button>
      </div>

      {tab === "document" && (
        <div className="tab-content">
          <DocSelector documents={documents} selectedDoc={activeDoc} onSelect={setActiveDoc} />
          <button className="primary-btn" onClick={checkDocumentInteractions} disabled={loading}>
            Check All Medications in Document
          </button>

          {loading && <LoadingSpinner text="Checking interactions..." />}

          {result && !loading && (
            <div className="interaction-results">
              {/* Summary */}
              <div className="summary-cards">
                <div className="summary-card">
                  <span className="summary-num">{result.total_medications}</span>
                  <span>Medications</span>
                </div>
                <div className="summary-card">
                  <span className="summary-num">{result.pairs_checked}</span>
                  <span>Pairs Checked</span>
                </div>
                <div className="summary-card danger">
                  <span className="summary-num">{result.severity_summary?.major || 0}</span>
                  <span>Major</span>
                </div>
                <div className="summary-card warning">
                  <span className="summary-num">{result.severity_summary?.moderate || 0}</span>
                  <span>Moderate</span>
                </div>
              </div>

              {/* Recommendation */}
              <div className={`recommendation ${result.severity_summary?.major > 0 ? "danger" : result.severity_summary?.moderate > 0 ? "warning" : "safe"}`}>
                {result.recommendation}
              </div>

              {/* Interactions */}
              {result.interactions?.map((interaction, i) => {
                const cfg = SEVERITY_CONFIG[interaction.severity] || SEVERITY_CONFIG.minor;
                return (
                  <div key={i} className="interaction-card" style={{ borderLeft: `4px solid ${cfg.color}` }}>
                    <div className="interaction-header">
                      <span>{cfg.icon} {interaction.drug_a} + {interaction.drug_b}</span>
                      <span className="severity-badge" style={{ background: cfg.color }}>
                        {cfg.label}
                      </span>
                    </div>
                    <p className="interaction-effect">{interaction.effect}</p>
                    <p className="interaction-management">💊 {interaction.management}</p>
                  </div>
                );
              })}

              {result.interactions?.length === 0 && (
                <div className="no-interactions">
                  <CheckCircle size={24} color="#66bb6a" />
                  <p>No significant interactions found</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {tab === "pair" && (
        <div className="tab-content">
          <div className="pair-check-form">
            <input
              className="drug-input"
              placeholder="Drug A (e.g. warfarin)"
              value={drugA}
              onChange={(e) => setDrugA(e.target.value)}
            />
            <span className="plus">+</span>
            <input
              className="drug-input"
              placeholder="Drug B (e.g. aspirin)"
              value={drugB}
              onChange={(e) => setDrugB(e.target.value)}
            />
            <button className="primary-btn" onClick={checkPair} disabled={loading}>
              Check
            </button>
          </div>

          {loading && <LoadingSpinner text="Checking..." />}

          {pairResult && !loading && (
            <div className="pair-result">
              {pairResult.interaction_found ? (
                <div
                  className="interaction-card"
                  style={{
                    borderLeft: `4px solid ${SEVERITY_CONFIG[pairResult.severity]?.color}`
                  }}
                >
                  <div className="interaction-header">
                    <span>
                      {SEVERITY_CONFIG[pairResult.severity]?.icon}{" "}
                      {pairResult.drug_a} + {pairResult.drug_b}
                    </span>
                    <span
                      className="severity-badge"
                      style={{ background: SEVERITY_CONFIG[pairResult.severity]?.color }}
                    >
                      {pairResult.severity?.toUpperCase()}
                    </span>
                  </div>
                  <p><strong>Mechanism:</strong> {pairResult.mechanism}</p>
                  <p><strong>Effect:</strong> {pairResult.effect}</p>
                  <p><strong>Management:</strong> {pairResult.management}</p>
                  <p className="source-label">Source: {pairResult.source}</p>
                </div>
              ) : (
                <div className="no-interactions">
                  <CheckCircle size={24} color="#66bb6a" />
                  <p>No significant interaction found between {drugA} and {drugB}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}