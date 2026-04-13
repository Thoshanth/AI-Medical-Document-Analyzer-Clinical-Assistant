export default function DocSelector({ documents, selectedDoc, onSelect }) {
  if (!documents.length) {
    return (
      <div className="no-doc-warning">
        No documents found. Upload a document first.
      </div>
    );
  }

  return (
    <div className="doc-selector">
      <label>Active Document</label>
      <select
        value={selectedDoc?.id || ""}
        onChange={(e) => {
          const doc = documents.find(
            (d) => d.id === parseInt(e.target.value)
          );
          onSelect(doc);
        }}
      >
        <option value="">Select a document...</option>
        {documents.map((doc) => (
          <option key={doc.id} value={doc.id}>
            [{doc.document_type}] {doc.filename}
          </option>
        ))}
      </select>
    </div>
  );
}