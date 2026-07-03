import type { AnswerResult } from "../types";

// One strategy's result: metrics row, answer, and cited-highlighted sources.
export function ResultPanel({ result }: { result: AnswerResult }) {
  const { trace, chunks, citations, answer, strategy } = result;
  const cited = new Set(citations);

  return (
    <div className="panel">
      <h2>{strategy}</h2>
      <div className="metrics">
        <span>retrieval <b>{trace.retrieval_ms} ms</b></span>
        <span>generation <b>{trace.generation_ms} ms</b></span>
        <span>total <b>{trace.total_ms} ms</b></span>
        <span>~tokens <b>{trace.approx_prompt_tokens}+{trace.approx_completion_tokens}</b></span>
        {trace.cache_hit && <span className="cache">cache hit ✓</span>}
      </div>

      <div className="answer">{answer}</div>

      <div className="sources">
        {chunks.map((c, i) => {
          const n = i + 1;
          const isCited = cited.has(n);
          return (
            <details key={c.id} open={isCited}>
              <summary>
                [Source {n}] <span className="src-tag">{c.source}</span>
                {isCited && <span className="cited"> ✓ cited</span>}
              </summary>
              <p>{c.text.slice(0, 500)}…</p>
            </details>
          );
        })}
      </div>
    </div>
  );
}
