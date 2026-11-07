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

// Narrow worker client entry for browser LiteRT-LM consumers. Excludes the
// Transformers.js 4 pipeline in the broad package barrel.
export { LiteRtLmWorkerClient } from './worker/client'
export type {
  LiteRtLmWorkerLoadOptions,
  LiteRtLmWorkerGenerationConfig,
  LiteRtLmWorkerPrompt,
  LiteRtLmWorkerPromptPart,
} from './worker/protocol'
