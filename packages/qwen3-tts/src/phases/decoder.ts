import type {
  RuntimeContext,
  PipelineProgress,
  ModelManifest,
} from '@lil-local/inference-core';
import { InferenceError } from '@lil-local/inference-core';
import { CodecDecoder } from '../codec';
import { discoverCodecShapes } from '../shape-discovery';
import {
  createQwen3TtsManifest,
  type Qwen3TtsVariant,
} from '../manifest';
import {
  unpackCodecFrames,
  type CodecFrames,
} from '../codec-frames';

export interface DecoderPhaseOptions {
  onProgress?: (progress: PipelineProgress) => void;
}

export class DecoderPhase {
  readonly manifest: ModelManifest;
  readonly name = 'decoder' as const;
  loadMs = 0;
  compileMs = 0;
  inferenceMs = 0;

  private readonly variant: Qwen3TtsVariant;
  private readonly onProgress?: (progress: PipelineProgress) => void;
  private context?: RuntimeContext;
  private codec?: CodecDecoder;

  constructor(
    variant: Qwen3TtsVariant,
    options: DecoderPhaseOptions = {},
  ) {
    this.variant = variant;
    this.onProgress = options.onProgress;
    this.manifest = createQwen3TtsManifest(variant);
  }

  async load(context: RuntimeContext): Promise<void> {
    this.context = context;
    const loadStart = performance.now();
    this.report({ phase: 'loading', step: 0, total: 1 });
    const compileStart = performance.now();
    const codecModel = await context.liteRt.loadModel(this.variant.codec);
    this.compileMs = performance.now() - compileStart;
    const codecShapes = discoverCodecShapes(codecModel);
    this.codec = new CodecDecoder(codecModel, {
      chunkSize: codecShapes.chunkSize,
      accelerator: this.context!.backend === 'webgpu' ? 'webgpu' : 'wasm',
    });
    this.loadMs = performance.now() - loadStart;
  }

  async decode(
    frames: CodecFrames,
    signal?: AbortSignal,
  ): Promise<Float32Array> {
    if (signal?.aborted) {
      throw new InferenceError(
        'CANCELLED',
        'Cancelled before decode',
        { stage: 'decoder' },
      );
    }
    const inferenceStart = performance.now();
    this.report({ phase: 'codec', step: 0, total: 1 });
    try {
      const allFrames = unpackCodecFrames(frames);
      const audio = await this.codec!.decode(allFrames);
      this.inferenceMs = performance.now() - inferenceStart;
      return audio;
    } catch (e) {
      throw e instanceof InferenceError
        ? e
        : new InferenceError('INFERENCE_FAILED', String(e), { cause: e });
    }
  }

  dispose(): void {
    this.codec = undefined;
    this.context = undefined;
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
