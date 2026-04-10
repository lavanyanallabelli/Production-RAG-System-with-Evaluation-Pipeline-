// src/App.js
import { useState } from "react";
import { Upload, MessageCircle, BarChart2, FileText, Loader } from "lucide-react";

const API = "http://localhost:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState("upload");

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <FileText size={24} color="#6366f1" />
        <h1 style={styles.title}>RAG System</h1>
        <p style={styles.subtitle}>Document Q&A with Hybrid Retrieval</p>
      </div>

      {/* Tabs */}
      <div style={styles.tabs}>
        {[
          { id: "upload", label: "Upload", icon: <Upload size={16} /> },
          { id: "ask", label: "Ask", icon: <MessageCircle size={16} /> },
          { id: "eval", label: "Eval", icon: <BarChart2 size={16} /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...styles.tab,
              ...(activeTab === tab.id ? styles.tabActive : {}),
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={styles.content}>
        {activeTab === "upload" && <UploadTab />}
        {activeTab === "ask" && <AskTab />}
        {activeTab === "eval" && <EvalTab />}
      </div>
    </div>
  );
}


// ── Tab 1: Upload ──────────────────────────────────────────────
function UploadTab() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.detail);
      setResult(data);

    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.cardTitle}>Upload PDF</h2>
      <p style={styles.hint}>Upload a PDF to build the knowledge base</p>

      <div style={styles.uploadBox}>
        <Upload size={32} color="#6366f1" />
        <p style={styles.uploadText}>
          {file ? file.name : "Choose a PDF file"}
        </p>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
          style={styles.fileInput}
        />
      </div>

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{
          ...styles.button,
          opacity: !file || loading ? 0.6 : 1,
        }}
      >
        {loading ? <><Loader size={16} /> Processing...</> : "Upload & Process"}
      </button>

      {result && (
        <div style={styles.successBox}>
          <p>✓ {result.message}</p>
          <p style={styles.meta}>Chunks created: {result.chunks_created}</p>
          <p style={styles.meta}>File: {result.filename}</p>
        </div>
      )}

      {error && <div style={styles.errorBox}>✕ {error}</div>}
    </div>
  );
}


// ── Tab 2: Ask ─────────────────────────────────────────────────
function AskTab() {
  const [question, setQuestion] = useState("");
  const [version, setVersion] = useState("v4");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, prompt_version: version }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      setResult(data);

    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = {
    high: "#22c55e",
    medium: "#f59e0b",
    low: "#ef4444",
    none: "#6b7280",
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.cardTitle}>Ask a Question</h2>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask anything about your document..."
        style={styles.textarea}
        rows={3}
      />

      <div style={styles.row}>
        <div>
          <label style={styles.label}>Prompt Version</label>
          <select
            value={version}
            onChange={(e) => setVersion(e.target.value)}
            style={styles.select}
          >
            {["v1", "v2", "v3", "v4"].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <button
          onClick={handleAsk}
          disabled={!question.trim() || loading}
          style={{
            ...styles.button,
            opacity: !question.trim() || loading ? 0.6 : 1,
          }}
        >
          {loading ? <><Loader size={16} /> Thinking...</> : "Ask"}
        </button>
      </div>

      {result && (
        <div style={styles.resultBox}>
          <div style={styles.answerHeader}>
            <span style={styles.answerLabel}>Answer</span>
            <span style={{
              ...styles.badge,
              backgroundColor: confidenceColor[result.confidence] + "20",
              color: confidenceColor[result.confidence],
            }}>
              {result.confidence} confidence
            </span>
          </div>

          <p style={styles.answer}>{result.answer}</p>

          {result.source_quote && (
            <div style={styles.quoteBox}>
              <p style={styles.quoteLabel}>Source</p>
              <p style={styles.quote}>"{result.source_quote}"</p>
            </div>
          )}

          <div style={styles.metaRow}>
            <span style={styles.meta}>Version: {result.prompt_version}</span>
            <span style={styles.meta}>Chunks used: {result.chunks_used}</span>
            <span style={styles.meta}>Retries: {result.retries}</span>
          </div>
        </div>
      )}

      {error && <div style={styles.errorBox}>✕ {error}</div>}
    </div>
  );
}


// ── Tab 3: Eval ────────────────────────────────────────────────
function EvalTab() {
  const [questions, setQuestions] = useState([
    { question: "", expected: "" }
  ]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const addQuestion = () => {
    setQuestions([...questions, { question: "", expected: "" }]);
  };

  const updateQuestion = (index, field, value) => {
    const updated = questions.map((q, i) =>
      i === index ? { ...q, [field]: value } : q
    );
    setQuestions(updated);
  };

  const removeQuestion = (index) => {
    setQuestions(questions.filter((_, i) => i !== index));
  };

  const handleEval = async () => {
    const valid = questions.filter(q => q.question.trim() && q.expected.trim());
    if (!valid.length) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API}/eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ test_questions: valid }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      setResult(data);

    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.cardTitle}>Evaluation Pipeline</h2>
      <p style={styles.hint}>Compare prompt versions with test questions</p>

      {questions.map((q, i) => (
        <div key={i} style={styles.questionCard}>
          <div style={styles.questionHeader}>
            <span style={styles.label}>Question {i + 1}</span>
            {questions.length > 1 && (
              <button
                onClick={() => removeQuestion(i)}
                style={styles.removeBtn}
              >
                ✕
              </button>
            )}
          </div>

          <input
            placeholder="Question..."
            value={q.question}
            onChange={(e) => updateQuestion(i, "question", e.target.value)}
            style={styles.input}
          />
          <input
            placeholder="Expected answer..."
            value={q.expected}
            onChange={(e) => updateQuestion(i, "expected", e.target.value)}
            style={{ ...styles.input, marginTop: 8 }}
          />
        </div>
      ))}

      <div style={styles.row}>
        <button onClick={addQuestion} style={styles.secondaryButton}>
          + Add Question
        </button>

        <button
          onClick={handleEval}
          disabled={loading}
          style={{ ...styles.button, opacity: loading ? 0.6 : 1 }}
        >
          {loading ? <><Loader size={16} /> Running Evals...</> : "Run Eval"}
        </button>
      </div>

      {result && (
        <div style={styles.resultBox}>
          <p style={styles.winnerText}>
            Winner: <strong>{result.winner}</strong>
          </p>

          <table style={styles.table}>
            <thead>
              <tr>
                {["Version", "Precision", "Faithfulness", "No Hallucination", "Retries"].map(h => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.comparison.map((row) => (
                <tr
                  key={row.version}
                  style={{
                    backgroundColor: row.version === result.winner ? "#f0fdf4" : "white",
                  }}
                >
                  <td style={styles.td}>
                    {row.version}
                    {row.version === result.winner && " 🏆"}
                  </td>
                  <td style={styles.td}>{(row.avg_precision * 100).toFixed(1)}%</td>
                  <td style={styles.td}>{(row.avg_faithfulness * 100).toFixed(1)}%</td>
                  <td style={styles.td}>{(row.low_hallucination_rate * 100).toFixed(1)}%</td>
                  <td style={styles.td}>{row.avg_retries.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {error && <div style={styles.errorBox}>✕ {error}</div>}
    </div>
  );
}


// ── Styles ─────────────────────────────────────────────────────
const styles = {
  container: {
    maxWidth: 800,
    margin: "0 auto",
    padding: "32px 16px",
    fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
    color: "#1e293b",
  },
  header: {
    textAlign: "center",
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    margin: "8px 0 4px",
  },
  subtitle: {
    color: "#64748b",
    margin: 0,
  },
  tabs: {
    display: "flex",
    gap: 8,
    marginBottom: 24,
    borderBottom: "1px solid #e2e8f0",
    paddingBottom: 0,
  },
  tab: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "10px 20px",
    border: "none",
    background: "none",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
    color: "#64748b",
    borderBottom: "2px solid transparent",
    marginBottom: -1,
  },
  tabActive: {
    color: "#6366f1",
    borderBottom: "2px solid #6366f1",
  },
  content: {
    marginTop: 8,
  },
  card: {
    background: "white",
    border: "1px solid #e2e8f0",
    borderRadius: 12,
    padding: 24,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 600,
    marginTop: 0,
    marginBottom: 4,
  },
  hint: {
    color: "#64748b",
    fontSize: 14,
    marginBottom: 20,
    marginTop: 0,
  },
  uploadBox: {
    border: "2px dashed #e2e8f0",
    borderRadius: 8,
    padding: 32,
    textAlign: "center",
    marginBottom: 16,
    position: "relative",
    cursor: "pointer",
  },
  uploadText: {
    color: "#64748b",
    marginTop: 8,
    marginBottom: 12,
  },
  fileInput: {
    position: "absolute",
    inset: 0,
    opacity: 0,
    cursor: "pointer",
    width: "100%",
    height: "100%",
  },
  button: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "10px 20px",
    background: "#6366f1",
    color: "white",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
  },
  secondaryButton: {
    padding: "10px 20px",
    background: "white",
    color: "#6366f1",
    border: "1px solid #6366f1",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
  },
  successBox: {
    background: "#f0fdf4",
    border: "1px solid #86efac",
    borderRadius: 8,
    padding: 16,
    marginTop: 16,
    color: "#166534",
  },
  errorBox: {
    background: "#fef2f2",
    border: "1px solid #fca5a5",
    borderRadius: 8,
    padding: 16,
    marginTop: 16,
    color: "#991b1b",
  },
  textarea: {
    width: "100%",
    padding: 12,
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    fontSize: 14,
    resize: "vertical",
    marginBottom: 12,
    boxSizing: "border-box",
    fontFamily: "inherit",
  },
  input: {
    width: "100%",
    padding: 10,
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    fontSize: 14,
    boxSizing: "border-box",
    fontFamily: "inherit",
  },
  select: {
    padding: "10px 12px",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    fontSize: 14,
    background: "white",
  },
  row: {
    display: "flex",
    gap: 12,
    alignItems: "flex-end",
    flexWrap: "wrap",
  },
  label: {
    display: "block",
    fontSize: 13,
    fontWeight: 500,
    color: "#475569",
    marginBottom: 4,
  },
  resultBox: {
    marginTop: 20,
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    padding: 16,
  },
  answerHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  answerLabel: {
    fontWeight: 600,
    fontSize: 14,
  },
  badge: {
    padding: "2px 10px",
    borderRadius: 99,
    fontSize: 12,
    fontWeight: 500,
  },
  answer: {
    lineHeight: 1.6,
    marginBottom: 12,
  },
  quoteBox: {
    background: "#f8fafc",
    borderLeft: "3px solid #6366f1",
    padding: "8px 12px",
    marginBottom: 12,
    borderRadius: "0 4px 4px 0",
  },
  quoteLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: "#6366f1",
    marginBottom: 4,
    marginTop: 0,
    textTransform: "uppercase",
  },
  quote: {
    color: "#475569",
    fontSize: 13,
    fontStyle: "italic",
    margin: 0,
  },
  metaRow: {
    display: "flex",
    gap: 16,
    flexWrap: "wrap",
  },
  meta: {
    fontSize: 12,
    color: "#94a3b8",
  },
  questionCard: {
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
  },
  questionHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  removeBtn: {
    background: "none",
    border: "none",
    color: "#94a3b8",
    cursor: "pointer",
    fontSize: 14,
  },
  winnerText: {
    marginTop: 0,
    marginBottom: 16,
    fontSize: 15,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 13,
  },
  th: {
    textAlign: "left",
    padding: "8px 12px",
    background: "#f8fafc",
    borderBottom: "1px solid #e2e8f0",
    fontWeight: 600,
    color: "#475569",
  },
  td: {
    padding: "8px 12px",
    borderBottom: "1px solid #f1f5f9",
  },
};