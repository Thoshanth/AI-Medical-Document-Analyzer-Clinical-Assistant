import {
  FileText, MessageSquare, ClipboardList,
  Pill, Bot, BarChart2, Activity
} from "lucide-react";

const navItems = [
  { id: "documents", label: "Documents", icon: FileText },
  { id: "chat", label: "Medical Chat", icon: MessageSquare },
  { id: "reports", label: "Reports", icon: ClipboardList },
  { id: "drugcheck", label: "Drug Safety", icon: Pill },
  { id: "agents", label: "Clinical Agents", icon: Bot },
  { id: "evaluation", label: "Evaluation", icon: BarChart2 },
];

export default function Sidebar({ activePage, onNavigate, selectedDoc }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Activity size={22} color="#4fc3f7" />
        <span>MedAI Platform</span>
      </div>

      {selectedDoc && (
        <div className="sidebar-active-doc">
          <span className="dot" />
          <span className="doc-name">{selectedDoc.filename}</span>
        </div>
      )}

      <nav className="sidebar-nav">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-item ${activePage === id ? "active" : ""}`}
            onClick={() => onNavigate(id)}
          >
            <Icon size={18} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span>AI Medical Platform v1.0</span>
        <span>11 Stages Active</span>
      </div>
    </aside>
  );
}