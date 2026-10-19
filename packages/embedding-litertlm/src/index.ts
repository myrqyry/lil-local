import type { BrowserSubstrate, EmbeddingInferenceResult } from '@lil-local/inference-core'
import type { EmbeddingEngineSettings } from '@litert-lm/core'

/**
 * Immutable source revision observed from the LiteRT Community Hub metadata
 * on 2026-10-18. This is a source pin, NOT a verified weight checksum or
 * successful browser inference receipt. Nothing downloads until load().
 */
export const EMBEDDING_GEMMA2_REVISION = '24d962e906c7d332c6428e71c9676855024569e2'
export const EMBEDDING_GEMMA2_MODEL_URL =
  `https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/${EMBEDDING_GEMMA2_REVISION}/embeddinggemma-2-740m.litertlm`

/** The browser-specific prefixes documented for LiteRT-LM EmbeddingGemma 2. */
export type EmbeddingGemma2Task =
  | 'search-query' | 'document' | 'classification' | 'clustering' | 'sentence-similarity'

const PREFIX: Record<EmbeddingGemma2Task, string> = {
  'search-query': 'task: search query | text: ',
  document: 'task: search result | text: ',
  classification: 'task: classification | text: ',
  clustering: 'task: clustering | text: ',
  'sentence-similarity': 'task: sentence similarity | text: ',
}

export function formatEmbeddingGemma2Text(text: string, task: EmbeddingGemma2Task): string {
  if (typeof text !== 'string' || !text.trim()) throw new Error('Embedding text must be nonempty')
  if (!Object.prototype.hasOwnProperty.call(PREFIX, task)) {
    throw new Error('Unsupported EmbeddingGemma 2 task')
  }
  return PREFIX[task] + text.trim()
}

export type EmbeddingGemma2Dimensions = 128 | 256 | 512 | 768

export interface EmbeddingGemma2Input {
  /** Supply the correct task prefix yourself, or use formatEmbeddingGemma2Text(). */
  text: string
  dimensions?: EmbeddingGemma2Dimensions
}

const LITERT_LM_0181_WASM_BASE = 'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.1/wasm'

export interface EmbeddingGemma2Options {
  /** Caller-owned source. No default model download or mutable Hugging Face reference. */
  model: EmbeddingEngineSettings['model']
  /** Optional locally hosted 0.18.1 WASM/JS base. Omit to use pinned jsDelivr. */
  wasmBase?: string
  /** Optional input length configuration; the native default is 1024. */
  maxInputLength?: number
}

function aborted(): Error {
  const error = new Error('EmbeddingGemma 2 model load aborted')
  error.name = 'AbortError'
  return error
}

/**
 * Explicit text-only LiteRT-LM EmbeddingGemma 2 adapter.
 *
 * Real EmbeddingEngine API, but NOT a browser model qualification.
 * Never loads a model on import or discovery. The caller provides its source
 * and owns version/integrity selection. Vision/audio encoders are not enabled.
 */
export function createEmbeddingGemma2Substrate(
  options: EmbeddingGemma2Options,
): BrowserSubstrate<EmbeddingGemma2Input, EmbeddingInferenceResult> {
  if (!options.model) throw new Error('EmbeddingGemma 2 requires an explicit model source')
  if (options.maxInputLength !== undefined &&
      (!Number.isSafeInteger(options.maxInputLength) || options.maxInputLength <= 0)) {
    throw new Error('maxInputLength must be a positive integer')
  }

  return {
    id: 'litert-lm:embeddinggemma-2-text',
    format: 'litertlm',
    capabilities: ['text-embedding'],
    requestedBackends: ['webgpu'],
    async create(_onProgress, signal) {
      if (signal?.aborted) throw aborted()
      const { EmbeddingEngine, Backend, getOrLoadGlobalLiteRtLm } = await import('@litert-lm/core')
      if (signal?.aborted) throw aborted()
      let model = options.model
      // Download only on explicit load, using a streaming response rather than
      // materializing the whole checkpoint. AbortSignal cancels the HTTP fetch.
      if (typeof model === 'string' && /^https?:\/\//i.test(model)) {
        const response = await fetch(model, { signal })
        if (!response.ok) throw new Error(`EmbeddingGemma 2 fetch failed (HTTP ${response.status})`)
        if (!response.body) throw new Error('EmbeddingGemma 2 fetch has no response body')
        model = response.body
      }
      if (signal?.aborted) throw aborted()
      // The upstream 0.18.1 SDK currently defaults to 0.12.1 WASM assets.
      // Explicitly require a version-matched path before any engine is created.
      await getOrLoadGlobalLiteRtLm(options.wasmBase ?? LITERT_LM_0181_WASM_BASE)
      if (signal?.aborted) throw aborted()
      const engine = await EmbeddingEngine.create({
        model,
        backend: Backend.GPU,
        maxInputLength: options.maxInputLength,
      })
      return {
        async infer(input) {
          if (typeof input?.text !== 'string' || !input.text.trim()) {
            throw new Error('Embedding text must be nonempty')
          }
          const dimensions = input.dimensions ?? 768
          if (![128, 256, 512, 768].includes(dimensions)) {
            throw new Error('Unsupported EmbeddingGemma 2 output dimension')
          }
          const result = await engine.computeEmbedding(input.text, {
            normalize: true,
            outputSize: dimensions,
          })
          if (!Array.isArray(result.embedding) || result.embedding.length !== dimensions) {
            throw new Error('EmbeddingGemma 2 returned unexpected dimension')
          }
          if (!result.embedding.every(value => typeof value === 'number' && Number.isFinite(value))) {
            throw new Error('EmbeddingGemma 2 returned non-finite values')
          }
          return {
            kind: 'embedding',
            dimensions,
            values: Float32Array.from(result.embedding),
          }
        },
        dispose: () => engine.delete(),
      }
    },
  }
}
