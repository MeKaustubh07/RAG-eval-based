import { useEffect, useState } from "react";
import { compare, getHealth } from "./api";
import { ResultPanel } from "./components/ResultPanel";
import type { CompareResponse } from "./types";

export function App() {
  const [strategies, setStrategies] = useState<string[]>([]);
  const [chunks, setChunks] = useState<number>(0);
  const [question, setQuestion] = useState("");
  const [stratA, setStratA] = useState("dense");
  const [stratB, setStratB] = useState("rerank");
  const [provider, setProvider] = useState("ollama");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => { setStrategies(h.strategies); setChunks(h.chunks); })
      .catch((e) => setError(String(e)));
  }, []);

  async function run() {
    if (question.trim().length < 3) return;
    setLoading(true); setError(null); setResult(null);
    try {
      setResult(await compare(question, stratA, stratB, provider));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>RAG Platform — <span>A/B Retrieval Comparison</span></h1>
        <div className="sub">
          Same question, two strategies, side by side · {chunks} chunks indexed
        </div>
      </header>

      <div className="controls">
        <label>Question
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="e.g. does creatine cause dehydration?"
          />
        </label>
        <label>Strategy A
          <select value={stratA} onChange={(e) => setStratA(e.target.value)}>
            {strategies.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label>Strategy B
          <select value={stratB} onChange={(e) => setStratB(e.target.value)}>
            {strategies.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label>Provider
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            <option value="ollama">ollama (local)</option>
            <option value="gemini">gemini</option>
          </select>
        </label>
        <button onClick={run} disabled={loading}>
          {loading ? "Running…" : "Compare"}
        </button>
      </div>

      {error && <div className="error">Error: {error}</div>}

      <div className="grid">
        {result ? (
          <>
            <ResultPanel result={result.a} />
            <ResultPanel result={result.b} />
          </>
        ) : (
          <div className="empty">Ask a question to compare two strategies.</div>
        )}
      </div>
    </div>
  );
}
