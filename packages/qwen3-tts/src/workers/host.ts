import { type InferencePhaseReceipt, InferenceError } from '@lil-local/inference-core';
import type { CodecFrames } from '../codec-frames';
import type { QwenTtsInput, QwenTtsConfig } from '../types';
import type { Qwen3TtsVariant } from '../manifest';
import type {
  GeneratorWorkerRequest,
  GeneratorWorkerResponse,
  DecoderWorkerRequest,
  DecoderWorkerResponse,
} from './protocol';

export interface GeneratorOutcome {
  frames: CodecFrames;
  phaseReceipt: InferencePhaseReceipt;
}

export interface DecoderOutcome {
  audio: Float32Array;
  phaseReceipt: InferencePhaseReceipt;
}

export type ProgressHandler = (progress: { phase: string; step: number; total: number }) => void;

function waitFor<T extends { type: string }>(
  worker: Worker,
  expectType: string,
  requestId?: number,
  onProgress?: ProgressHandler,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<T>) => {
      const msg = event.data;
      if (msg.type === 'progress') {
        const progress = (msg as unknown as { progress: { phase: string; step: number; total: number } }).progress;
        onProgress?.(progress);
        return;
      }
      if (msg.type === 'error') {
        const err = (msg as unknown as { error: { code?: string; message: string } }).error;
        reject(new InferenceError(err.code === 'CANCELLED' ? 'CANCELLED' : 'INFERENCE_FAILED', err.message));
        return;
      }
      if (msg.type === expectType && (requestId === undefined || (msg as unknown as { requestId?: number }).requestId === requestId)) {
        resolve(msg);
      }
    };
    worker.onerror = (event) => {
      reject(new InferenceError('INFERENCE_FAILED', `worker error: ${event.message}`));
    };
  });
}

export async function runHostGenerator(
  worker: Worker,
  variant: Qwen3TtsVariant,
  modelBase: string,
  input: QwenTtsInput,
  config: QwenTtsConfig,
  onProgress?: ProgressHandler,
): Promise<GeneratorOutcome> {
  await waitFor<GeneratorWorkerResponse>(worker, 'booted');
  const init = waitFor<GeneratorWorkerResponse>(worker, 'ready');
  worker.postMessage({ type: 'initialize', variant, modelBase } satisfies GeneratorWorkerRequest);
  await init;
  const framesWait = waitFor<GeneratorWorkerResponse>(worker, 'frames', 1, onProgress);
  worker.postMessage({ type: 'generate', requestId: 1, input, config } satisfies GeneratorWorkerRequest);
  const msg = (await framesWait) as Extract<GeneratorWorkerResponse, { type: 'frames' }>;
  return { frames: msg.frames, phaseReceipt: msg.phaseReceipt };
}

export async function runHostDecoder(
  worker: Worker,
  variant: Qwen3TtsVariant,
  modelBase: string,
  frames: CodecFrames,
  onProgress?: ProgressHandler,
): Promise<DecoderOutcome> {
  await waitFor<DecoderWorkerResponse>(worker, 'booted');
  const init = waitFor<DecoderWorkerResponse>(worker, 'ready');
  worker.postMessage({ type: 'initialize', variant, modelBase } satisfies DecoderWorkerRequest);
  await init;
  const audioWait = waitFor<DecoderWorkerResponse>(worker, 'audio', 1, onProgress);
  worker.postMessage({ type: 'decode', requestId: 1, frames } satisfies DecoderWorkerRequest, [frames.frames.buffer]);
  const msg = (await audioWait) as Extract<DecoderWorkerResponse, { type: 'audio' }>;
  return { audio: msg.audio, phaseReceipt: msg.phaseReceipt };
}
