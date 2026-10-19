import {
  type AudioInferenceResult,
  type InferenceReceipt,
  type Pipeline,
  type PipelineProgress,
  type PipelineStatus,
  type RuntimeContext,
  createInferenceReceipt,
} from '@lil-local/inference-core';
import { checkAudioValid } from '@lil-local/inference-core';
import { kokoroManifest } from "./manifest";

export interface KokoroInput {
  text: string;
}

export interface KokoroConfig {
  voice?: string;
  speed?: number;
}

const DEFAULTS: KokoroConfig = {
  voice: "af_heart",
  speed: 1,
};

interface KokoroTtsLike {
  generate(
    text: string,
    config?: { voice?: string; speed?: number }
  ): Promise<{ audio: Float32Array; sampling_rate: number }>;
  list_voices(): void;
}

export class KokoroPipeline
  implements Pipeline<KokoroInput, AudioInferenceResult, KokoroConfig>
{
  readonly manifest = kokoroManifest;
  status: PipelineStatus = "idle";
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
  private tts: KokoroTtsLike | null = null;
  private loadMs = 0;

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === "ready") return;
    this.status = "loading";
    this.context = context;
    const loadStart = performance.now();
    try {
      this.report({ phase: "loading", step: 1, total: 2 });
      const { KokoroTTS } = await import("kokoro-js");
      this.report({ phase: "loading", step: 2, total: 2 });
      this.tts = (await KokoroTTS.from_pretrained(
        "onnx-community/Kokoro-82M-v1.0-ONNX",
        { dtype: "q8", device: "wasm" }
      )) as unknown as KokoroTtsLike;
      this.loadMs = performance.now() - loadStart;
      this.status = "ready";
    } catch (e) {
      this.status = "error";
      throw e instanceof Error
        ? (e as Error & { code?: string })
        : new Error(String(e));
    }
  }

  async run(
    input: KokoroInput,
    config?: KokoroConfig,
    signal?: AbortSignal
  ): Promise<AudioInferenceResult> {
    if (this.status !== "ready") {
      throw new Error("Pipeline not ready");
    }
    if (!this.tts) {
      throw new Error("Kokoro TTS not loaded");
    }
    this.status = "running";
    const cfg = { ...DEFAULTS, ...config };
    const inferenceStart = performance.now();
    try {
      if (signal?.aborted) throw new Error("CANCELLED");
      const audio = await this.tts.generate(input.text, {
        voice: cfg.voice,
        speed: cfg.speed,
      });
      if (signal?.aborted) throw new Error("CANCELLED");
      const samples = audio.audio;
      const sampleRate = audio.sampling_rate || 24000;
      const duration = samples.length / sampleRate;
      const warnings = checkAudioValid(samples, sampleRate, 1, duration);
      if (warnings.length > 0) {
        console.warn("[KokoroPipeline]", ...warnings);
      }
      this.status = "ready";
      return {
        kind: "audio",
        samples,
        sampleRate,
        channels: 1,
        durationSeconds: duration,
          receipt: createInferenceReceipt({
            manifest: this.manifest,
            backend: this.context?.backend ?? 'wasm',
            loadMs: this.loadMs,
            compileMs: 0,
            inferenceStart,
            inputSummary: `${input.text.length} characters`,
            outputSummary: `${samples.length} samples at ${sampleRate}Hz, 1 channel`,
            warnings,
          }),
      };
    } catch (e) {
      this.status = "ready";
      throw e instanceof Error
        ? (e as Error & { code?: string })
        : new Error(String(e));
    }
  }

  async dispose(): Promise<void> {
    this.tts = null;
    this.context = null;
    this.status = "disposed";
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
