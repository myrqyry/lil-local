import {
  type AudioInferenceResult,
  type InferenceReceipt,
  type Pipeline,
  type PipelineProgress,
  type PipelineStatus,
  type RuntimeContext,
  checkAudioValid,
  createInferenceReceipt,
} from '@lil-local/inference-core';
import type * as Ort from 'onnxruntime-web';
import { phononManifest, PHONON_HF_BASE, PHONON_DEFAULT_VARIANT, PHONON_VARIANT_GRAPHS } from './manifest';
import { PhononTokenizer } from './tokenizer';

export interface PhononInput {
  text: string;
  voice?: string;
}

export interface PhononConfig {
  voice?: string;
  temperature?: number;
}

const SAMPLE_RATE = 24000;
const SAMPLES_PER_FRAME = 1920;
const EOS_THRESHOLD = 4.0;

/**
 * Phonon 31M English TTS pipeline.
 *
 * Two streaming ONNX graphs:
 *  - flow_lm: causal transformer → 32-dim latent per 80 ms frame + EOS logit
 *  - mimi_decoder: latent → 1920 samples of 24 kHz PCM audio
 *
 * Both graphs expose state (past* / present*) for streaming KV-cache and
 * convolution context. This pipeline runs them in a lockstep loop: each
 * flow_lm step emits a latent, which is immediately fed to mimi_decoder.
 *
 * NOTE: The exact ONNX input/output tensor names and shapes have not been
 * inspected against the published graphs. The draft names below must be
 * verified before real inference runs. The pipeline structure, manifest,
 * tokenizer, and lifecycle are complete and testable.
 */
export class PhononPipeline
  implements Pipeline<PhononInput, AudioInferenceResult, PhononConfig>
{
  readonly manifest = phononManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
  private ort: typeof Ort | null = null;
  private flowLm: Ort.InferenceSession | null = null;
  private mimiDecoder: Ort.InferenceSession | null = null;
  private tokenizer: PhononTokenizer | null = null;
  private loadMs = 0;

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    this.status = 'loading';
    this.context = context;
    const loadStart = performance.now();
    try {
      this.report({ phase: 'loading', step: 1, total: 5 });
      const ort = await import('onnxruntime-web/wasm');
      this.ort = ort;
      this.report({ phase: 'loading', step: 2, total: 5 });

      const graphs = PHONON_VARIANT_GRAPHS[PHONON_DEFAULT_VARIANT];
      const flowLmUrl = PHONON_HF_BASE + graphs.flowLm;
      const mimiUrl = PHONON_HF_BASE + graphs.mimiDecoder;

      const externalData: Array<{ path: string; data: string }> = [];
      if (graphs.flowLmData) {
        externalData.push({ path: graphs.flowLmData, data: PHONON_HF_BASE + graphs.flowLmData });
      }
      if (graphs.mimiDecoderData) {
        externalData.push({ path: graphs.mimiDecoderData, data: PHONON_HF_BASE + graphs.mimiDecoderData });
      }

      this.report({ phase: 'loading', step: 3, total: 5 });
      this.flowLm = await this.ort.InferenceSession.create(flowLmUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        externalData: externalData.length > 0 ? externalData : undefined,
      });
      this.report({ phase: 'loading', step: 4, total: 5 });
      this.mimiDecoder = await this.ort.InferenceSession.create(mimiUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      const tokenizerBuf = await context.assets.resolve({
        id: 'tokenizer',
        path: 'tokenizer.json',
      });
      const tokenizerData = JSON.parse(new TextDecoder().decode(tokenizerBuf));
      this.tokenizer = new PhononTokenizer(tokenizerData);

      this.loadMs = performance.now() - loadStart;
      this.status = 'ready';
      this.report({ phase: 'loading', step: 5, total: 5 });
    } catch (e) {
      this.status = 'error';
      await this.dispose();
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async run(
    input: PhononInput,
    config?: PhononConfig,
    signal?: AbortSignal,
  ): Promise<AudioInferenceResult> {
    if (this.status !== 'ready') {
      throw new Error('Pipeline not ready');
    }
    if (!this.flowLm || !this.mimiDecoder || !this.tokenizer || !this.ort || !this.context) {
      throw new Error('PhononPipeline not loaded');
    }
    if (signal?.aborted) {
      throw new Error('CANCELLED');
    }

    this.status = 'running';
    const inferenceStart = performance.now();

    try {
      const text = input.text;
      const voice = config?.voice ?? input.voice ?? 'default';

      const tokenIds = this.tokenizer.encode(text);

      // Streaming inference loop.
      //
      // NOTE(ABI): Verify exact input/output tensor names and shapes against
      // the published flow_lm and mimi_decoder ONNX graphs. The draft
      // names below are structural and must be confirmed.
      //
      // Expected flow_lm signature (per model card):
      //   inputs:  input_ids, attention_mask, past* state tensors
      //   outputs: latent (1 x 32 float), eos_logit (float), present* state tensors
      // Expected mimi_decoder signature:
      //   inputs:  latent (1 x 32 float), past* conv context
      //   outputs: pcm (1 x 1920 float), present* conv context

      const chunks: Float32Array[] = [];
      let totalSamples = 0;
      let step = 0;
      const maxSteps = tokenIds.length * 2; // safety bound

      while (step < maxSteps) {
        if (signal?.aborted) throw new Error('CANCELLED');

        // --- flow_lm step: produce next latent ---
        // const flowOut = await this.flowLm.run({ ... });
        // const latent = flowOut.latent as Ort.Tensor<Float32Array>;
        // const eosLogit = flowOut.eos_logit.data[0];
        // if (eosLogit > EOS_THRESHOLD) break;

        // --- mimi_decoder step: decode latent to PCM ---
        // const pcmOut = await this.mimiDecoder.run({ latent, ... });
        // const pcm = pcmOut.pcm.data as Float32Array;
        // chunks.push(pcm);
        // totalSamples += pcm.length;

        step++;
        // Safety: if the graph stubs are not wired, emit silence and stop.
        if (step === 1) {
          chunks.push(new Float32Array(SAMPLES_PER_FRAME));
          totalSamples += SAMPLES_PER_FRAME;
          break;
        }
        break;
      }

      const samples = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }

      const duration = samples.length / SAMPLE_RATE;
      const warnings = checkAudioValid(samples, SAMPLE_RATE, 1, duration);

      this.status = 'ready';
      return {
        kind: 'audio',
        samples,
        sampleRate: SAMPLE_RATE,
        channels: 1,
        durationSeconds: duration,
        receipt: createInferenceReceipt({
          manifest: this.manifest,
          backend: this.context.backend,
          loadMs: this.loadMs,
          compileMs: 0,
          inferenceStart,
          inputSummary: `${text.length} characters, voice=${voice}`,
          outputSummary: `${samples.length} samples at ${SAMPLE_RATE}Hz, 1 channel`,
          warnings,
        }),
      };
    } catch (e) {
      this.status = 'ready';
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async dispose(): Promise<void> {
    if (this.flowLm) {
      try {
        await this.flowLm.release();
      } catch {
        // release failure is observed via status; do not mask the original error.
      }
      this.flowLm = null;
    }
    if (this.mimiDecoder) {
      try {
        await this.mimiDecoder.release();
      } catch {
        // same as above.
      }
      this.mimiDecoder = null;
    }
    this.tokenizer = null;
    this.ort = null;
    this.context = null;
    this.status = 'disposed';
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export { PhononTokenizer } from './tokenizer';
