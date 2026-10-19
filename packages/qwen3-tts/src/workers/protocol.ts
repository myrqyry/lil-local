import type { InferencePhaseReceipt } from '@lil-local/inference-core';
import type { CodecFrames } from '../codec-frames';
import type { Qwen3TtsVariant } from '../manifest';
import type { QwenTtsConfig } from '../types';

export interface SerializedInferenceError {
  code: string;
  message: string;
  stage?: string;
}

export function serializeError(e: unknown): SerializedInferenceError {
  if (typeof e === 'object' && e !== null && 'code' in e && 'message' in e) {
    const as = e as { code?: string; message?: string; stage?: string };
    return { code: as.code ?? 'UNKNOWN', message: String(as.message), stage: as.stage };
  }
  if (e instanceof Error) return { code: 'UNKNOWN', message: e.message };
  return { code: 'UNKNOWN', message: String(e) };
}

export type GeneratorWorkerRequest =
  | { type: 'initialize'; variant: Qwen3TtsVariant; modelBase: string }
  | { type: 'generate'; requestId: number; input: { text: string }; config: QwenTtsConfig }
  | { type: 'cancel'; requestId: number };

export type GeneratorWorkerResponse =
  | { type: 'booted' }
  | { type: 'ready' }
  | { type: 'progress'; requestId?: number; progress: { phase: string; step: number; total: number } }
  | { type: 'frames'; requestId: number; frames: CodecFrames; phaseReceipt: InferencePhaseReceipt }
  | { type: 'error'; requestId?: number; error: SerializedInferenceError };

export type DecoderWorkerRequest =
  | { type: 'initialize'; variant: Qwen3TtsVariant; modelBase: string }
  | { type: 'decode'; requestId: number; frames: CodecFrames };

export type DecoderWorkerResponse =
  | { type: 'booted' }
  | { type: 'ready' }
  | { type: 'audio'; requestId: number; audio: Float32Array; phaseReceipt: InferencePhaseReceipt }
  | { type: 'error'; requestId?: number; error: SerializedInferenceError };
