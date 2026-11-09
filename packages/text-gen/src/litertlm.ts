// Narrow browser LiteRT-LM consumer entrypoint. Keep the Transformers.js
// pipeline out of bundlers that only need native LiteRT-LM inference.
export { LiteRtLmTextPipeline, resolveTextGenerationManifest } from './litertlm-pipeline'
export type {
  LiteRtLmBackend,
  LiteRtLmModelSource,
  LiteRtLmTextConfig,
  LiteRtLmTextPipelineOptions,
  LiteRtLmToolCall,
} from './litertlm-pipeline'
export type { TextMessage, TextGenerationConfig, TextGenerationInput } from './types'
export { gemma4E2bManifest, gemma4E2bWebManifest, gemma4E4bWebManifest } from './manifest'
export { getPinnedGemma4WebSource, getPinnedGemma4FullSource } from './gemma4-web'

// Narrow worker client entry for browser LiteRT-LM consumers. Excludes the
// Transformers.js 4 pipeline in the broad package barrel.
import { LiteRtLmWorkerClient } from './worker/client'
export { LiteRtLmWorkerClient }
export type {
  LiteRtLmWorkerLoadOptions,
  LiteRtLmWorkerGenerationConfig,
  LiteRtLmWorkerPrompt,
  LiteRtLmWorkerPromptPart,
  LiteRtLmWorkerStats,
} from './worker/protocol'

/**
 * LiteRT-LM's @litertjs/wasm-utils loader invokes importScripts(), which
 * cannot run in an ES-module worker. Vite emits the worker as a classic IIFE
 * here so native LiteRT-LM Wasm initialization can work in a worker thread.
 *
 * The default client remains an ES-module worker for consumers that need it.
 */
export function createClassicLiteRtLmWorkerClient(): LiteRtLmWorkerClient {
  return new LiteRtLmWorkerClient(
    () => new Worker(new URL('./worker/litertlm.worker.ts', import.meta.url)),
  )
}
