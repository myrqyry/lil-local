import type { ManagedLiteRtRuntime } from '@lil-local/runtime-litert';

import type { ManagedLiteRtRuntimeContext } from '@lil-local/runtime-litert';
import {
  decodeMoGeOutputs,
  MOGE_MODEL_ID,
  MOGE_SIZE,
  prepareMoGeImage,
  type MoGeGeometry,
  type MoGeTensorData,
} from './mogeGeometry';

// Immutable Hugging Face commit, not the mutable main branch.
export const MOGE_REVISION = '6af93bb77c766c9c63aa364e51506a98cc35ccba';
const MOGE_BASE = 'https://huggingface.co/' + MOGE_MODEL_ID + '/resolve/' + MOGE_REVISION;
export const MOGE_GPU_URL = MOGE_BASE + '/moge_fp16.tflite';
export const MOGE_CPU_URL = MOGE_BASE + '/moge.tflite';

export interface MoGeEstimateOptions {
  signal?: AbortSignal;
  frameId?: string | number;
  /** Explicit backend disables automatic GPU -> WASM fallback. */
  backend?: 'webgpu' | 'wasm';
  /** In automatic mode, permit the larger FP32 CPU model if GPU compile fails. */
  allowCpuFallback?: boolean;
  onProgress?: (progress: {loadedBytes: number; totalBytes?: number}) => void;
}

function cancelled(error: unknown, signal?: AbortSignal): boolean {
  return signal?.aborted === true || (error instanceof Error && error.name === 'AbortError') ||
    (typeof error === 'object' && error !== null &&
      'code' in error && (error as {code: unknown}).code === 'CANCELLED');
}

function backendUnavailable(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error &&
    ((error as {code: unknown}).code === 'BACKEND_UNAVAILABLE' ||
     (error as {code: unknown}).code === 'MODEL_COMPILE_FAILED');
}

async function runOnBackend(
  runtime: ManagedLiteRtRuntime,
  image: ImageData,
  values: Float32Array,
  backend: 'webgpu' | 'wasm',
  options: MoGeEstimateOptions,
): Promise<MoGeGeometry> {
  const url = backend === 'webgpu' ? MOGE_GPU_URL : MOGE_CPU_URL;
  const loadOptions = {accelerator: backend, signal: options.signal, onProgress: options.onProgress} as const;
  const model = await runtime.loadModel(url, loadOptions);
  const details = model.getInputDetails();
  if (details.length !== 1 ||
      Array.from(details[0].shape).join(',') !== [1, 3, MOGE_SIZE, MOGE_SIZE].join(',') ||
      String(details[0].dtype) !== 'float32') {
    throw new Error('MoGe-2 LiteRT input must be float32 NCHW 1x3x448x448');
  }
  const outputDetails = model.getOutputDetails();
  const input = runtime.createTensor(values, [1, 3, MOGE_SIZE, MOGE_SIZE]);
  let outputs: Awaited<ReturnType<typeof runtime.predict>> | undefined;
  try {
    const started = performance.now();
    const inferenceResult = await runtime.predict(url, input, {...loadOptions, label: 'moge-2-geometry'});
    outputs = inferenceResult;
    const inferenceMs = performance.now() - started;
    const tensors = Array.isArray(inferenceResult)
      ? inferenceResult
      : outputDetails.map((detail) => {
          const tensor = inferenceResult[detail.name];
          if (!tensor) throw new Error('MoGe-2 output ' + detail.name + ' is missing');
          return tensor;
        });
    if (tensors.length !== outputDetails.length) {
      throw new Error('MoGe-2 output count changed');
    }
    const raw: MoGeTensorData[] = [];
    for (let index = 0; index < tensors.length; index += 1) {
      const detail = outputDetails[index];
      if (String(detail.dtype) !== 'float32') throw new Error('MoGe-2 output dtype changed');
      const tensor = tensors[index];
      // WebGPU tensors cannot be read synchronously. LiteRT.js moveTo('wasm')
      // performs the supported asynchronous GPU -> CPU readback.
      const readable = backend === 'webgpu' ? await tensor.moveTo('wasm') : tensor;
      try {
        // Copy before releasing any WASM tensor (typed views may alias its memory).
        raw.push({
          name: detail.name,
          shape: Array.from(detail.shape),
          values: runtime.readTensor<Float32Array>(readable).slice(),
        });
      } finally {
        if (readable !== tensor) readable.delete();
      }
    }
    return decodeMoGeOutputs(raw, {
      modelId: MOGE_MODEL_ID,
      modelRevision: MOGE_REVISION,
      backend,
      sourceWidth: image.width,
      sourceHeight: image.height,
      frameId: options.frameId,
      inferenceMs,
    });
  } finally {
    // Do not hold GPU tensors across future geometry calls.
    if (outputs) {
      const tensors = Array.isArray(outputs) ? outputs : Object.values(outputs);
      for (const tensor of new Set(tensors)) tensor.delete();
    }
    input.delete();
  }
}

/**
 * Real MoGe-2 inference through a caller-owned shared Lil Local runtime.
 * Lazy: downloads the model only when explicitly called.
 *
 * Affine 3D points are NOT metric coordinates. The returned relativeDepth
 * is normalized per frame; it must not be used for real-world measurements
 * or game-agent collision guarantees.
 */
export async function estimateMoGeGeometry(
  context: Pick<ManagedLiteRtRuntimeContext, 'liteRt' | 'backend'>,
  image: ImageData,
  options: MoGeEstimateOptions = {},
): Promise<MoGeGeometry> {
  if (options.signal?.aborted) throw new DOMException('MoGe-2 cancelled', 'AbortError');
  const values = prepareMoGeImage(image);
  const runtime = context.liteRt;
  const primary = options.backend ?? (context.backend === 'webgpu' ? 'webgpu' : 'wasm');
  try {
    return await runOnBackend(runtime, image, values, primary, options);
  } catch (error) {
    if (cancelled(error, options.signal)) throw error;
    if (options.backend !== undefined || options.allowCpuFallback === false ||
        primary !== 'webgpu' || !backendUnavailable(error)) {
      throw error;
    }
    return runOnBackend(runtime, image, values, 'wasm', options);
  }
}

/** Release cached MoGe models without disposing the shared browser inference runtime. */
export function releaseMoGeModels(runtime: ManagedLiteRtRuntime): void {
  runtime.disposeModel(MOGE_GPU_URL);
  runtime.disposeModel(MOGE_CPU_URL);
}
