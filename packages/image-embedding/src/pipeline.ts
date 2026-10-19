import { Tensor } from '@litertjs/core';
import { InferenceError } from '@lil-local/inference-core';
import type {
  ModelManifest,
  Pipeline,
  PipelineProgress,
  PipelineStatus,
  RuntimeContext,
} from '@lil-local/inference-core';
import type { ManagedLiteRtRuntime } from '@lil-local/runtime-litert';
import { clipImageEmbeddingManifest } from './manifest';

const INPUT_SIZE = 224;
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

export interface ClipTextEmbeddings {
  count: number;
  dimension: number;
  embeddings: Float32Array;
}

export interface ClipLabelResult {
  label: string;
  index: number;
  score: number;
  probability: number;
}

export interface ClipImageInput {
  canvas: HTMLCanvasElement | OffscreenCanvas;
}

export interface ClipImageConfig {
  topK?: number;
}

export interface ClipImageOutput {
  kind: 'embedding';
  values: Float32Array;
  dimensions: number;
  ranked: ClipLabelResult[];
}

export function decodeClipTextEmbeddings(buffer: ArrayBuffer): ClipTextEmbeddings {
  if (buffer.byteLength < 8) throw new Error('Invalid CLIP text embeddings header');

  const view = new DataView(buffer);
  const count = view.getInt32(0, true);
  const dimension = view.getInt32(4, true);
  const expectedBytes = 8 + count * dimension * Float32Array.BYTES_PER_ELEMENT;
  if (count <= 0 || dimension <= 0 || expectedBytes !== buffer.byteLength) {
    throw new Error('Invalid CLIP text embeddings dimensions');
  }

  return {
    count,
    dimension,
    embeddings: new Float32Array(buffer.slice(8)),
  };
}

export function rankClipLabels(
  imageEmbedding: Float32Array,
  textEmbeddings: Float32Array,
  labels: string[],
  topK = labels.length,
): ClipLabelResult[] {
  if (textEmbeddings.length % imageEmbedding.length !== 0) {
    throw new Error('CLIP image and text embedding dimensions do not match');
  }

  const dimension = imageEmbedding.length;
  const imageNorm = Math.sqrt(imageEmbedding.reduce((sum, value) => sum + value * value, 0)) || 1;
  const scores: number[] = [];
  for (let offset = 0; offset < textEmbeddings.length; offset += dimension) {
    let dot = 0;
    let textNorm = 0;
    for (let i = 0; i < dimension; i++) {
      const value = textEmbeddings[offset + i];
      dot += imageEmbedding[i] * value;
      textNorm += value * value;
    }
    scores.push((dot / imageNorm / (Math.sqrt(textNorm) || 1)) * 100);
  }

  const max = Math.max(...scores);
  const probabilities = scores.map((score) => Math.exp(score - max));
  const total = probabilities.reduce((sum, value) => sum + value, 0) || 1;
  return scores
    .map((score, index) => ({
      label: labels[index] ?? `class_${index}`,
      index,
      score,
      probability: probabilities[index] / total,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(0, topK));
}

export class ClipImageEmbeddingPipeline implements Pipeline<ClipImageInput, ClipImageOutput, ClipImageConfig> {
  readonly manifest: ModelManifest = clipImageEmbeddingManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
  private runtime: ManagedLiteRtRuntime | null = null;
  private labels: string[] = [];
  private textEmbeddings: ClipTextEmbeddings | null = null;

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    this.context = context;
    this.status = 'loading';
    this.report({ phase: 'loading', step: 0, total: 1 });
    try {
      const runtime = context.liteRt as unknown as ManagedLiteRtRuntime;
      const modelAsset = this.manifest.assets.find(a => a.id === 'model');
      const textAsset = this.manifest.assets.find(a => a.id === 'text-embeddings');
      const labelsAsset = this.manifest.assets.find(a => a.id === 'labels');
      if (!modelAsset || !textAsset || !labelsAsset) {
        throw new InferenceError('INFERENCE_FAILED', 'Missing CLIP assets');
      }
      await runtime.loadModel(modelAsset.path);
      const [embeddingsBuffer, labelsBuffer] = await Promise.all([
        context.assets.resolve(textAsset),
        context.assets.resolve(labelsAsset),
      ]);
      this.textEmbeddings = decodeClipTextEmbeddings(embeddingsBuffer);
      this.labels = new TextDecoder().decode(labelsBuffer)
        .split(/\r?\n/)
        .map(label => label.trim())
        .filter(Boolean);
      if (this.labels.length !== this.textEmbeddings.count) {
        throw new InferenceError('INFERENCE_FAILED', `CLIP label count mismatch: ${this.labels.length} != ${this.textEmbeddings.count}`);
      }
      this.runtime = runtime;
      this.status = 'ready';
      this.report({ phase: 'loading', step: 1, total: 1 });
    } catch (e) {
      this.status = 'error';
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e });
    }
  }

  async run(input: ClipImageInput, config?: ClipImageConfig, signal?: AbortSignal): Promise<ClipImageOutput> {
    if (this.status !== 'ready' || !this.runtime || !this.textEmbeddings) {
      throw new InferenceError('INFERENCE_FAILED', 'Pipeline not ready');
    }
    this.status = 'running';
    const cfg = { topK: config?.topK ?? 5 };
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const modelAsset = this.manifest.assets.find(a => a.id === 'model')!;
      const tensor = this.canvasToTensor(input.canvas);
      try {
        const rawOutput = await this.runtime.predict(modelAsset.path, tensor, { label: 'clip-image-embedding' });
        const outputs = Array.isArray(rawOutput) ? rawOutput : Object.values(rawOutput);
        try {
          const imageEmbedding = new Float32Array(outputs[0].toTypedArray());
          const embeddings = this.textEmbeddings;
          const result: ClipImageOutput = {
            kind: 'embedding',
            values: imageEmbedding,
            dimensions: imageEmbedding.length,
            ranked: rankClipLabels(imageEmbedding, embeddings.embeddings, this.labels, cfg.topK),
          };
          this.status = 'ready';
          return result;
        } finally {
          outputs.forEach((output) => output.delete());
        }
      } finally {
        tensor.delete();
      }
    } catch (e) {
      this.status = 'ready';
      throw e;
    }
  }

  async dispose(): Promise<void> {
    if (this.runtime) {
      const modelAsset = this.manifest.assets.find(a => a.id === 'model');
      if (modelAsset) this.runtime.disposeModel(modelAsset.path);
    }
    this.runtime = null;
    this.context = null;
    this.labels = [];
    this.textEmbeddings = null;
    this.status = 'disposed';
  }

  private canvasToTensor(canvas: HTMLCanvasElement | OffscreenCanvas): Tensor {
    const sourceWidth = canvas.width;
    const sourceHeight = canvas.height;
    const cropSize = Math.min(sourceWidth, sourceHeight);
    const sourceX = (sourceWidth - cropSize) / 2;
    const sourceY = (sourceHeight - cropSize) / 2;
    const target = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE)
      : document.createElement('canvas');
    target.width = INPUT_SIZE;
    target.height = INPUT_SIZE;
    const context = target.getContext('2d');
    if (!context) throw new Error('Unable to create CLIP preprocessing context');
    context.drawImage(canvas, sourceX, sourceY, cropSize, cropSize, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const data = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const pixels = INPUT_SIZE * INPUT_SIZE;
    const values = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      const offset = i * 4;
      for (let channel = 0; channel < 3; channel++) {
        values[channel * pixels + i] = (data[offset + channel] / 255 - CLIP_MEAN[channel]) / CLIP_STD[channel];
      }
    }
    return Tensor.fromTypedArray(values, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
