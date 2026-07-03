// Thin typed API client. In dev, Vite proxies these paths to :8000.

import type { CompareResponse, Health } from "./types";

export async function getHealth(): Promise<Health> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`/health ${res.status}`);
  return res.json();
}

export async function compare(
  question: string,
  strategyA: string,
  strategyB: string,
  provider: string,
): Promise<CompareResponse> {
  const res = await fetch("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      strategy_a: strategyA,
      strategy_b: strategyB,
      provider,
      k: 5,
    }),
  });
  if (!res.ok) throw new Error(`/compare ${res.status}`);
  return res.json();
}
