import { useState, useEffect } from "react";
import {
  runEvaluation, testEmergencyDetection,
  getEvaluationResults, getDocuments,
} from "../api";
import DocSelector from "../components/DocSelector";
import LoadingSpinner from "../components/LoadingSpinner";
import { CheckCircle, XCircle, BarChart2 } from "lucide-react";

const MetricBar = ({ label, value, max = 1 }) => {
  const pct = Math.round((value / max) * 100);
  const color = pct >= 80 ? "#66bb6a" : pct >= 60 ? "#ffa726" : "#ef5350";
  return (
    <div className="metric-bar">
      <div className="metric-label">
        <span>{label}</span>
        <span style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="metric-track">
        <div
          className="metric-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
};

export default function EvaluationPage({ selectedDoc: propDoc, showToast }) {
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(propDoc);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [result, setResult] = useState(null);
  const [emergencyResult, setEmergencyResult] = useState(null);

  useEffect(() => {
    getDocuments().then(setDocuments).catch(() => {});
  }, []);

  useEffect(() => { setActiveDoc(propDoc); }, [propDoc]);

  const runFullEval = async () => {
    if (!activeDoc) { showToast("Select a document first", "error"); return; }
    setLoading(true);
    setLoadingMsg("Running full evaluation (3-5 minutes)...");
    try {
      const data = await runEvaluation(activeDoc.id);
      setResult(data);
      showToast(`✅ Evaluation complete: ${data.overall_grade}`, "success");
    } catch {
      showToast("Evaluation failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const runEmergencyTest = async () => {
    setLoading(true);
    setLoadingMsg("Testing emergency detection...");
    try {
      const data = await testEmergencyDetection();
      setEmergencyResult(data);
      showToast(`Emergency detection: ${(data.accuracy * 100).toFixed(0)}% accuracy`, "success");
    } catch {
      showToast("Emergency test failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const gradeColor = (grade) => {
    if (!grade) return "#78909c";
    if (grade.startsWith("A")) return "#66bb6a";
    if (grade.startsWith("B")) return "#4fc3f7";
    if (grade.startsWith("C")) return "#ffa726";
    return "#ef5350";
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Medical AI Evaluation</h1>
        <p>RAGAS metrics + medical-specific quality assessment</p>
      </div>

      <div className="eval-controls">
        <DocSelector documents={documents} selectedDoc={activeDoc} onSelect={setActiveDoc} />
        <div className="eval-buttons">
          <button className="primary-btn" onClick={runFullEval} disabled={loading}>
            Run Full Evaluation
          </button>
          <button className="secondary-btn" onClick={runEmergencyTest} disabled={loading}>
            Test Emergency Detection
          </button>
        </div>
      </div>

      {loading && <LoadingSpinner text={loadingMsg} />}

      {/* Emergency test results */}
      {emergencyResult && !loading && (
        <div className="eval-section">
          <h2>Emergency Detection Results</h2>
          <div className="metric-grid">
            <div className="metric-card">
              <span className="metric-value" style={{ color: emergencyResult.sensitivity >= 0.95 ? "#66bb6a" : "#ef5350" }}>
                {(emergencyResult.sensitivity * 100).toFixed(0)}%
              </span>
              <span>Sensitivity</span>
              <span className="metric-threshold">Required: ≥95%</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{(emergencyResult.specificity * 100).toFixed(0)}%</span>
              <span>Specificity</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{(emergencyResult.accuracy * 100).toFixed(0)}%</span>
              <span>Accuracy</span>
            </div>
          </div>

          <div className="test-cases">
            {emergencyResult.test_results?.map((t, i) => (
              <div key={i} className={`test-case ${t.correct ? "pass" : "fail"}`}>
                {t.correct
                  ? <CheckCircle size={16} color="#66bb6a" />
                  : <XCircle size={16} color="#ef5350" />
                }
                <span>{t.text}</span>
                <span className="expected">
                  Expected: {t.expected_emergency ? "🚨 Emergency" : "✅ Normal"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Full evaluation results */}
      {result && !loading && (
        <div className="eval-results">
          {/* Overall grade */}
          <div className="grade-card">
            <div
              className="grade-letter"
              style={{ color: gradeColor(result.overall_grade) }}
            >
              {result.overall_grade?.charAt(0)}
            </div>
            <div>
              <h2>{result.overall_grade}</h2>
              <p>Overall Score: {(result.overall_score * 100).toFixed(1)}%</p>
              <p className={result.production_ready ? "ready" : "not-ready"}>
                {result.production_ready
                  ? "✅ Production Ready"
                  : "⚠️ Not Yet Production Ready"
                }
              </p>
            </div>
          </div>

          {/* RAGAS metrics */}
          {result.ragas_metrics?.metrics && (
            <div className="eval-section">
              <h2>RAGAS Metrics</h2>
              <p className="section-desc">
                Grade: {result.ragas_metrics.grade}
              </p>
              <MetricBar label="Faithfulness (Anti-hallucination)" value={result.ragas_metrics.metrics.faithfulness} />
              <MetricBar label="Answer Relevancy" value={result.ragas_metrics.metrics.answer_relevancy} />
              <MetricBar label="Context Precision" value={result.ragas_metrics.metrics.context_precision} />
              <MetricBar label="Context Recall" value={result.ragas_metrics.metrics.context_recall} />
            </div>
          )}

          {/* Medical metrics */}
          <div className="eval-section">
            <h2>Medical Safety Metrics</h2>
            <div className="criteria-grid">
              {Object.entries(result.passing_criteria || {}).map(([key, pass]) => (
                <div key={key} className={`criteria-item ${pass ? "pass" : "fail"}`}>
                  {pass
                    ? <CheckCircle size={16} color="#66bb6a" />
                    : <XCircle size={16} color="#ef5350" />
                  }
                  <span>{key.replace(/_/g, " ")}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations */}
          {result.recommendations?.length > 0 && (
            <div className="eval-section">
              <h2>Recommendations</h2>
              {result.recommendations.map((rec, i) => (
                <div key={i} className="recommendation-item">
                  💡 {rec}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}