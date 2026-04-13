import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { runClinicalAgents, getDocuments } from "../api";
import DocSelector from "../components/DocSelector";
import LoadingSpinner from "../components/LoadingSpinner";
import { ChevronDown, ChevronRight } from "lucide-react";

const AGENT_ICONS = {
  Triage: "🚑",
  Diagnosis: "🔬",
  Pharmacist: "💊",
  Research: "📚",
  Safety: "🛡️",
};

export default function AgentsPage({ selectedDoc: propDoc, showToast }) {
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(propDoc);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showTrace, setShowTrace] = useState(false);
  const [expandedAgent, setExpandedAgent] = useState(null);

  useEffect(() => {
    getDocuments().then(setDocuments).catch(() => {});
  }, []);

  useEffect(() => { setActiveDoc(propDoc); }, [propDoc]);

  const runAgents = async () => {
    if (!activeDoc) { showToast("Select a document first", "error"); return; }
    if (!question.trim()) { showToast("Enter a clinical question", "error"); return; }
    setLoading(true);
    setResult(null);
    try {
      const data = await runClinicalAgents(activeDoc.id, question, showTrace);
      setResult(data);
      if (data.is_emergency) showToast("🚨 EMERGENCY detected!", "error");
      else showToast("✅ Clinical analysis complete", "success");
    } catch (err) {
      showToast("Agent analysis failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const urgencyColor = {
    emergency: "#ef5350",
    urgent: "#ffa726",
    routine: "#66bb6a",
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Clinical Agent System</h1>
        <p>5 specialized AI agents analyze your medical document</p>
      </div>

      {/* Agent icons */}
      <div className="agent-lineup">
        {Object.entries(AGENT_ICONS).map(([name, icon]) => (
          <div key={name} className="agent-chip">
            <span>{icon}</span>
            <span>{name}</span>
          </div>
        ))}
      </div>

      <div className="agents-form">
        <DocSelector documents={documents} selectedDoc={activeDoc} onSelect={setActiveDoc} />
        <textarea
          className="agent-question"
          placeholder="Clinical question (e.g. What is the full clinical assessment for this patient?)"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
        />
        <div className="agent-controls">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={showTrace}
              onChange={(e) => setShowTrace(e.target.checked)}
            />
            Show agent reasoning trace
          </label>
          <button className="primary-btn" onClick={runAgents} disabled={loading}>
            Run 5-Agent Analysis
          </button>
        </div>
      </div>

      {loading && (
        <LoadingSpinner text="5 clinical agents analyzing... (this takes 1-2 minutes)" />
      )}

      {result && !loading && (
        <div className="agent-results">
          {/* Status bar */}
          <div className="agent-status-bar">
            <span
              className="urgency-pill"
              style={{ background: urgencyColor[result.urgency_level] || "#555" }}
            >
              {result.urgency_level?.toUpperCase()}
            </span>
            <span>Primary: {result.primary_diagnosis || "Under investigation"}</span>
            <span>Iterations: {result.iterations_used}</span>
            <span className={result.safety_approved ? "approved" : "pending"}>
              {result.safety_approved ? "✅ Safety Approved" : "⏳ Pending"}
            </span>
          </div>

          {/* Critical alerts */}
          {result.critical_alerts?.length > 0 && (
            <div className="critical-section">
              <h3>🚨 Critical Alerts</h3>
              {result.critical_alerts.map((alert, i) => (
                <div key={i} className="critical-alert-item">{alert}</div>
              ))}
            </div>
          )}

          {/* Final answer */}
          <div className="final-answer">
            <h3>Clinical Assessment</h3>
            <ReactMarkdown>{result.final_answer}</ReactMarkdown>
          </div>

          {/* Agent trace */}
          {showTrace && result.agent_trace && (
            <div className="agent-trace">
              <h3>Agent Reasoning Trace</h3>
              {result.agent_trace.map((step, i) => (
                <div key={i} className="trace-item">
                  <button
                    className="trace-header"
                    onClick={() => setExpandedAgent(expandedAgent === i ? null : i)}
                  >
                    <span>
                      {AGENT_ICONS[step.agent] || "🤖"} {step.agent} Agent
                      — Iteration {step.iteration}
                    </span>
                    {expandedAgent === i
                      ? <ChevronDown size={16} />
                      : <ChevronRight size={16} />
                    }
                  </button>
                  {expandedAgent === i && (
                    <div className="trace-content">
                      {Object.entries(step)
                        .filter(([k]) => !["agent", "iteration"].includes(k))
                        .map(([k, v]) => (
                          <p key={k}>
                            <strong>{k}:</strong>{" "}
                            {typeof v === "boolean" ? v.toString() : v}
                          </p>
                        ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}