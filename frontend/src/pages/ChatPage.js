import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { medicalQuery, getDocuments } from "../api";
import DocSelector from "../components/DocSelector";
import { Send, Paperclip, AlertCircle } from "lucide-react";

export default function ChatPage({ selectedDoc: propDoc, showToast }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(propDoc);
  const bottomRef = useRef();
  const textRef = useRef();

  useEffect(() => {
    getDocuments().then(setDocuments).catch(() => {});
  }, []);

  useEffect(() => {
    setActiveDoc(propDoc);
  }, [propDoc]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMsg },
    ]);
    setLoading(true);

    try {
      const result = await medicalQuery(
        userMsg,
        activeDoc?.id || null,
        5
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
          safetyInfo: result.safety_info,
          criticalAlerts: result.critical_alerts,
        },
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail;
      let errorMsg = "Something went wrong.";

      if (typeof detail === "object") {
        errorMsg = `🚫 ${detail.error}\n\n${detail.reason || ""}\n\n${detail.suggestion || ""}`;
      } else if (typeof detail === "string") {
        errorMsg = `🚫 ${detail}`;
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorMsg, isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const quickQuestions = [
    "What are the key findings in this document?",
    "Are there any abnormal lab values?",
    "What medications are mentioned?",
    "What is the diagnosis?",
  ];

  return (
    <div className="page chat-page">
      <div className="page-header">
        <h1>Medical Chat</h1>
        <DocSelector
          documents={documents}
          selectedDoc={activeDoc}
          onSelect={setActiveDoc}
        />
      </div>

      {/* Messages */}
      <div className="messages-area">
        {messages.length === 0 && (
          <div className="welcome-screen">
            <div className="welcome-icon">⚕️</div>
            <h2>Medical AI Assistant</h2>
            <p>
              Ask questions about your medical documents.
              {activeDoc
                ? ` Searching in: ${activeDoc.filename}`
                : " Select a document or ask general medical questions."}
            </p>
            <div className="quick-questions">
              {quickQuestions.map((q) => (
                <button
                  key={q}
                  className="quick-btn"
                  onClick={() => { setInput(q); textRef.current?.focus(); }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === "assistant" && (
              <div className="avatar">⚕</div>
            )}
            <div className={`bubble ${msg.isError ? "error-bubble" : ""}`}>
              {/* Critical alerts */}
              {msg.criticalAlerts?.length > 0 && (
                <div className="critical-alerts">
                  {msg.criticalAlerts.map((a, j) => (
                    <div key={j} className="critical-alert">
                      <AlertCircle size={14} />
                      {a.finding || a}
                    </div>
                  ))}
                </div>
              )}

              <ReactMarkdown>{msg.content}</ReactMarkdown>

              {/* Sources */}
              {msg.sources?.length > 0 && (
                <div className="sources">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-tag">
                      📄 {s.filename} §{s.chunk_index}
                    </span>
                  ))}
                </div>
              )}

              {/* Safety info */}
              {msg.safetyInfo?.warnings?.length > 0 && (
                <div className="safety-warning">
                  ⚠️ {msg.safetyInfo.warnings[0]}
                </div>
              )}

              {/* Hallucination warning */}
              {msg.safetyInfo?.hallucination_risk === "high" && (
                <div className="hallucination-warning">
                  🔍 High hallucination risk — verify with source documents
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="avatar">⚕</div>
            <div className="bubble loading-bubble">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="input-box">
          <textarea
            ref={textRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a medical question..."
            rows={1}
            disabled={loading}
          />
          <button
            className={`send-btn ${input.trim() && !loading ? "active" : ""}`}
            onClick={sendMessage}
            disabled={!input.trim() || loading}
          >
            <Send size={18} />
          </button>
        </div>
        <p className="input-hint">
          Medical AI — Always consult a healthcare professional
        </p>
      </div>
    </div>
  );
}