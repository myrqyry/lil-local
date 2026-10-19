import type { EmbeddingInferenceResult, MultiVectorEmbeddingResult, RetrievalResult } from '@lil-local/inference-core'

export function dot(a: Float32Array, b: Float32Array, startA = 0, startB = 0, dim = a.length): number {
  let sum = 0
  for (let i = 0; i < dim; i++) sum += a[startA + i] * b[startB + i]
  return sum
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const dim = Math.min(a.length, b.length)
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < dim; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na * nb)
  return denom === 0 ? 0 : dot / denom
}

export function rankDense<T>(
  query: EmbeddingInferenceResult,
  candidates: Array<{ id: string; embedding: EmbeddingInferenceResult; payload?: T }>,
  topK?: number,
): Array<RetrievalResult<T>> {
  const scored = candidates
    .map((c) => ({ id: c.id, score: cosineSimilarity(query.values, c.embedding.values), payload: c.payload }))
    .sort((a, b) => b.score - a.score)
  return topK === undefined ? scored : scored.slice(0, topK)
}

export function maxSim(query: MultiVectorEmbeddingResult, document: MultiVectorEmbeddingResult): number {
  const qTokens = query.tokens
  const dTokens = document.tokens
  const dim = Math.min(query.dimensions, document.dimensions)
  let score = 0
  for (let q = 0; q < qTokens; q++) {
    let best = -Infinity
    for (let d = 0; d < dTokens; d++) {
      best = Math.max(best, dot(query.values, document.values, q * dim, d * dim, dim))
    }
    if (best === -Infinity) continue
    score += best
  }
  return score
}

export function rankColBert<T>(
  query: MultiVectorEmbeddingResult,
  candidates: Array<{ id: string; embedding: MultiVectorEmbeddingResult; payload?: T }>,
  topK?: number,
): Array<RetrievalResult<T>> {
  const scored = candidates
    .map((c) => ({ id: c.id, score: maxSim(query, c.embedding), payload: c.payload }))
    .sort((a, b) => b.score - a.score)
  return topK === undefined ? scored : scored.slice(0, topK)
}
