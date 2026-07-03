// Shared types mirroring the FastAPI response shapes (app.py).

export interface Trace {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  approx_prompt_tokens: number;
  approx_completion_tokens: number;
  cache_hit: boolean;
  cache_similarity?: number;
}

export interface Chunk {
  id: string;
  source: string;
  text: string;
  score: number;
}

export interface AnswerResult {
  question: string;
  strategy: string;
  provider: string;
  answer: string;
  citations: number[]; // 1-based source numbers the model cited
  cited_chunk_ids: string[];
  chunks: Chunk[];
  trace: Trace;
}

export interface CompareResponse {
  a: AnswerResult;
  b: AnswerResult;
}

export interface Health {
  status: string;
  chunks: number;
  strategies: string[];
}
