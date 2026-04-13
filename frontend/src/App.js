import { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import DocumentsPage from "./pages/DocumentsPage";
import ChatPage from "./pages/ChatPage";
import ReportsPage from "./pages/ReportsPage";
import DrugCheckPage from "./pages/DrugCheckPage";
import AgentsPage from "./pages/AgentsPage";
import EvaluationPage from "./pages/EvaluationPage";
import "./App.css";

export default function App() {
  const [activePage, setActivePage] = useState("documents");
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = "info") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  const pages = {
    documents: (
      <DocumentsPage
        documents={documents}
        setDocuments={setDocuments}
        selectedDoc={selectedDoc}
        setSelectedDoc={setSelectedDoc}
        showToast={showToast}
        onNavigate={setActivePage}
      />
    ),
    chat: (
      <ChatPage
        selectedDoc={selectedDoc}
        showToast={showToast}
      />
    ),
    reports: (
      <ReportsPage
        selectedDoc={selectedDoc}
        showToast={showToast}
      />
    ),
    drugcheck: (
      <DrugCheckPage
        selectedDoc={selectedDoc}
        showToast={showToast}
      />
    ),
    agents: (
      <AgentsPage
        selectedDoc={selectedDoc}
        showToast={showToast}
      />
    ),
    evaluation: (
      <EvaluationPage
        selectedDoc={selectedDoc}
        showToast={showToast}
      />
    ),
  };

  return (
    <div className="app">
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        selectedDoc={selectedDoc}
      />
      <main className="main-content">
        {pages[activePage]}
      </main>
    </div>
  );
}