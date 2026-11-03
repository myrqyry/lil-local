/**
 * Explicit browser content-safety classifier over the Transformers.js
 * text-classification pipeline. Separate from Lil LiteRT: this is an ONNX
 * ModernBERT encoder, not a LiteRT graph.
 *
 * No model is downloaded by module import, render, discovery, or preflight.
 * Nothing loads until the consumer explicitly calls load() on its runner.
 */
import type { BrowserSubstrate } from '@lil-local/inference-core'
import {
  MODERATION_MODEL_ID,
  MODERATION_MODEL_REVISION,
  decodeModerationScores,
  validateModerationInput,
  type ModerationInput,
  type ModerationResult,
} from './contract'

export type ModerationDevice = 'webgpu' | 'wasm'

/**
 * Caller-chosen Transformers.js dtype. No single variant is hard-coded; pick the
 * ONNX build that suits the target device. Valid values for this model are `q8`
 * (int8, ~141 MB — the browser-friendly default), `fp16` (~282 MB), `fp32`
 * (~563 MB) and `bnb4` (~418 MB).
 */
export type ModerationDtype = string

/** Load-progress shape forwarded to the caller. */
export interface ModerationProgress {
  status?: string
  progress?: number
  file?: string
}

export interface ModerationOptions {
  /** Execution backend to request. Defaults to `webgpu`. */
  device?: ModerationDevice
  /** ONNX dtype to load. Defaults to `q8`. */
  dtype?: ModerationDtype
  /** Optional load-progress callback forwarded to the Transformers.js loader. */
  onProgress?: (progress: ModerationProgress) => void
}

/** Local module shape; the package does not trust the library's own types at the boundary. */
interface TransformersModule {
  pipeline: (
    task: string,
    modelId: string,
    options?: {
      device?: string
      dtype?: string
      revision?: string
      progress_callback?: (progress: ModerationProgress) => void
    },
  ) => Promise<ModerationClassifier>
}

interface ModerationClassifier {
  (
    input: string | { text: string; text_pair?: string },
    options?: { top_k?: number | null },
  ): Promise<unknown>
  dispose?: () => Promise<void> | void
}

function aborted(): Error {
  const error = new Error('Content-safety-guard model load aborted')
  error.name = 'AbortError'
  return error
}

/**
 * Build a lazily-loading browser moderation substrate. The returned substrate
 * never touches the network until its `create()` is invoked by an explicit load;
 * wrap it in `BrowserSubstrateRunner` to own load/dispose races and cancellation.
 */
export function createModerationSubstrate(
  options: ModerationOptions = {},
): BrowserSubstrate<ModerationInput, ModerationResult> {
  const device = options.device ?? 'webgpu'
  const dtype = options.dtype ?? 'q8'

  return {
    id: 'transformers:onnx-content-safety-guard-small',
    format: 'onnx',
    capabilities: ['text-classification'],
    requestedBackends: [device],
    async create(_onProgress, signal) {
      if (signal?.aborted) throw aborted()
      const transformers = (await import('@huggingface/transformers')) as unknown as TransformersModule
      if (signal?.aborted) throw aborted()
      const classifier = await transformers.pipeline('text-classification', MODERATION_MODEL_ID, {
        device,
        dtype,
        revision: MODERATION_MODEL_REVISION,
        progress_callback: options.onProgress ?? (() => undefined),
      })
      if (signal?.aborted) {
        await classifier.dispose?.()
        throw aborted()
      }
      return {
        async infer(input) {
          const { text, textPair } = validateModerationInput(input)
          const call = textPair === undefined ? text : { text, text_pair: textPair }
          const raw = await classifier(call, { top_k: null })
          const { unsafe, categories } = decodeModerationScores(raw)
          return { kind: 'moderation', unsafe, categories }
        },
        async dispose() {
          await classifier.dispose?.()
        },
      }
    },
  }
}
