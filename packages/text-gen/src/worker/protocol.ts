export interface LiteRtLmWorkerMessage {
  role: 'user' | 'assistant' | 'model';
  content: string;
}

export interface LiteRtLmWorkerGenerationConfig {
  temperature?: number;
  maxOutputTokens?: number;
  topK?: number;
  topP?: number;
  seed?: number;
  maxContextTokens?: number;
  history?: LiteRtLmWorkerMessage[];
  systemPrompt?: string;
}

export type LiteRtLmWorkerEngine = 'litert-lm' | 'mediapipe';

export interface LiteRtLmWorkerMediaPipeLoadOptions {
  wasmBaseUrl?: string;
  maxTokens?: number;
  topK?: number;
  temperature?: number;
  randomSeed?: number;
  maxNumImages?: number;
  supportAudio?: boolean;
  disableRewinding?: boolean;
}

export interface LiteRtLmWorkerLoadOptions {
  engine?: LiteRtLmWorkerEngine;
  /** Self-hosted JS/WASM base for LiteRT-LM; defaults to a version-matched CDN. */
  wasmBase?: string;
  /** Opt-in for a compatible native vision model; the official -web Gemma 4 checkpoints have no functioning vision executor in JS 0.19.0. */
  visionBackend?: 'cpu' | 'gpu';
  /** Only a compatible native audio model; the known Gemma 4 -web checkpoints abort on WAV in JS 0.19.0. */
  audioBackend?: 'cpu' | 'gpu';
  /** Maximum concurrently attached images, matching LiteRT-LM EngineSettings. */
  maxNumImages?: number;
  mediaPipe?: LiteRtLmWorkerMediaPipeLoadOptions;
  /**
   * Optional caller-owned identity for Blob-backed models. Supply this when
   * independently-created Blob objects represent the same cached model and
   * should deduplicate an in-flight load.
   */
  loadKey?: string;
}

export type LiteRtLmWorkerPromptPart =
  | { type: 'text'; text: string }
  | { type: 'image'; data: Blob }
  /** SDK-native audio Blob, matching the official LiteRT-LM browser demo. */
  | { type: 'audio'; data: Blob }
  | {
      type: 'audio';
      /**
       * PCM samples remain caller-owned. The current client structured-clones
       * this array when posting to the worker, so large clips incur one buffer
       * copy rather than detaching the caller's data.
       */
      audioSamples: Float32Array;
      audioSampleRateHz: number;
    };

export type LiteRtLmWorkerPrompt = string | LiteRtLmWorkerPromptPart[];

export type LiteRtLmWorkerRequest =
  | { type: 'load'; model: string | Blob; options?: LiteRtLmWorkerLoadOptions }
  | {
      type: 'generate';
      id: string;
      prompt: LiteRtLmWorkerPrompt;
      config?: LiteRtLmWorkerGenerationConfig;
    }
  | { type: 'cancel'; id: string }
  | { type: 'dispose' };

export type LiteRtLmWorkerResponse =
  | { type: 'ready' }
  | { type: 'token'; id: string; text: string }
  | { type: 'reasoning'; id: string; text: string }
  | { type: 'complete'; id: string }
  | { type: 'cancelled'; id: string }
  | { type: 'error'; id?: string; message: string }
  | { type: 'disposed' };
