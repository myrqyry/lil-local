import {
  type AudioInferenceResult, type Pipeline, type PipelineProgress,
  type PipelineStatus, type RuntimeContext, checkAudioValid, createInferenceReceipt,
} from '@lil-local/inference-core';
import type * as Ort from 'onnxruntime-web';
import { PHONON_HF_BASE, PHONON_VARIANT_GRAPHS, phononManifest, manifestForPhononVariant, type PhononVariant } from './manifest';
import { choosePhononVariant } from './variant';

/** The published ONNX model uses this exact tensor ABI (verified from the graph files). */
const RATE = 24000;
const LATENT_DIM = 32;
const VOICE_DIM = 512;
const FRAME_RATE = 12.5;
const FLOW_LAYERS = 6;
const DECODER_LAYERS = 2;
const HEADS = 8;
const HEAD_DIM = 64;
const EOS_THRESHOLD = 4;
const CONV_STATES: Record<string, readonly number[]> = {
  'past_conv.upsample': [1, 512, 16],
  'past_conv.init_conv': [1, 512, 6],
  'past_conv.up0': [1, 256, 6],
  'past_conv.up0.res0.conv1': [1, 256, 2],
  'past_conv.up1': [1, 128, 5],
  'past_conv.up1.res0.conv1': [1, 128, 2],
  'past_conv.up2': [1, 64, 4],
  'past_conv.up2.res0.conv1': [1, 64, 2],
  'past_conv.final_conv': [1, 64, 2],
};
const TEXT_MODEL = 'onnx-community/phonon-31m-en-ONNX';

interface TokenizerLike {
  encode(text: string, options: { add_special_tokens: boolean }): number[];
  _tokenizer?: { model?: {
    byte_fallback?: boolean;
    encode?: (tokens: unknown) => string[];
    tokens_to_ids?: Map<string, number>;
  }};
}
type State = Record<string, Ort.Tensor>;

export type PhononRuntimeContext = Pick<RuntimeContext, 'backend' | 'assets' | 'signal'>;
export interface PhononInput { text: string; voice?: string }
export interface PhononConfig {
  voice?: string;
  temperature?: number;
  /** Receives real 24 kHz PCM frames as the decoder finishes each frame. */
  onChunk?: (samples: Float32Array) => void;
}
export interface PhononPipelineOptions {
  /** Model graph URLs and voice metadata live under this pinned origin by default. */
  modelBase?: string;
  /** Choose both graph precisions together. WASM typically prefers fp32/q4. */
  variant?: PhononVariant;
}

function requireOutput(output: Ort.InferenceSession.OnnxValueMapType, key: string): Ort.Tensor {
  const tensor = output[key];
  if (!tensor) throw new Error('Phonon graph missing output ' + key);
  return tensor as Ort.Tensor;
}

function stateFrom(output: Ort.InferenceSession.OnnxValueMapType): State {
  const state: State = {};
  for (const [name, value] of Object.entries(output)) {
    if (!name.startsWith('present.') && !name.startsWith('present_')) continue;
    const key = name.replace(/^present\./, 'past_key_values.').replace(/^present_/, 'past_');
    state[key] = value as Ort.Tensor;
  }
  return state;
}

/** ONNX Runtime GPU buffers are explicitly owned; JS garbage collection does not release them. */
function releaseGpu(...states: State[]): void {
  const seen = new Set<Ort.Tensor>();
  for (const state of states) for (const tensor of Object.values(state)) {
    if (!seen.has(tensor) && tensor.location === 'gpu-buffer') {
      seen.add(tensor);
      tensor.dispose();
    }
  }
}

function cache(ort: typeof Ort, layers: number): State {
  const result: State = {};
  for (let i = 0; i < layers; i++) for (const kind of ['key', 'value']) {
    result['past_key_values.' + i + '.' + kind] =
      new ort.Tensor('float32', new Float32Array(0), [1, HEADS, 0, HEAD_DIM]);
  }
  return result;
}

function randNormal(): number {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}
function checkSignal(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Error('CANCELLED');
}
function normalizeText(input: string): string {
  const text = input.trim().replace(/\s+/g, ' ')
    .replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, '-');
  if (!text) throw new Error('Phonon text cannot be empty');
  const capitalized = text[0].toUpperCase() + text.slice(1);
  return /[\p{L}\p{N}]$/u.test(capitalized) ? capitalized + '.' : capitalized;
}

/**
 * Real stateful ONNX graph execution. No fake PCM, empty caches, or test-only
 * tensor aliases: this follows the repository's two actual ONNX graph ABIs.
 */
export class PhononPipeline implements Pipeline<PhononInput, AudioInferenceResult, PhononConfig> {
  manifest = phononManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;
  private context: PhononRuntimeContext | null = null;
  private ort: typeof Ort | null = null;
  private flowLm: Ort.InferenceSession | null = null;
  private mimiDecoder: Ort.InferenceSession | null = null;
  private tokenizer: TokenizerLike | null = null;
  private loadMs = 0;
  private busy = false;
  private inFlight: Promise<void> | null = null;
  private disposing: Promise<void> | null = null;
  private readonly base: string;
  private readonly requestedVariant?: PhononVariant;

  constructor(options: PhononPipelineOptions = {}) {
    this.base = options.modelBase ?? PHONON_HF_BASE;
    this.requestedVariant = options.variant;
  }

  async load(context: PhononRuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    if (this.status !== 'idle') throw new Error('Phonon cannot load in ' + this.status + ' state');
    if (context.backend !== 'webgpu' && context.backend !== 'wasm')
      throw new Error('Phonon ONNX supports only WebGPU or WASM');
    this.status = 'loading';
    this.context = context;
    const start = performance.now();
    try {
      checkSignal(context.signal);
      const adapter = context.backend === 'webgpu' ? await navigator.gpu?.requestAdapter() : null;
      const variant = choosePhononVariant(context.backend, !!adapter?.features.has('shader-f16'), this.requestedVariant);
      const graphs = PHONON_VARIANT_GRAPHS[variant];
      this.manifest = manifestForPhononVariant(variant);
      const [{ AutoTokenizer }, ort] = await Promise.all([
        import('@huggingface/transformers'),
        context.backend === 'webgpu' ? import('onnxruntime-web/webgpu') : import('onnxruntime-web/wasm'),
      ]);
      this.ort = ort;
      this.onProgress?.({ phase: 'loading', step: 1, total: 4 });
      checkSignal(context.signal);
      const tokenizer = await AutoTokenizer.from_pretrained(TEXT_MODEL, {
        revision: PHONON_HF_BASE.match(/resolve\/([^/]+)\//)?.[1] ?? 'main',
      }) as unknown as TokenizerLike;
      // Transformers.js <= 4.3 still needs the published Unigram byte fallback patch.
      const model = tokenizer._tokenizer?.model;
      if (model && !model.byte_fallback && model.encode && model.tokens_to_ids) {
        const tokenize = model.encode.bind(model);
        const ids = model.tokens_to_ids;
        model.encode = (tokens) => tokenize(tokens).flatMap(piece => {
          if (ids.has(piece)) return [piece];
          const bytes = Array.from(new TextEncoder().encode(piece),
            b => '<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>');
          return bytes.every(b => ids.has(b)) ? bytes : [piece];
        });
      }
      this.tokenizer = tokenizer;
      this.onProgress?.({ phase: 'loading', step: 2, total: 4 });
      const provider = context.backend === 'webgpu' ? 'webgpu' : 'wasm';
      const gpu = provider === 'webgpu';
      const flowOutputs = ['latent_out', ...Array.from({ length: FLOW_LAYERS }, (_, i) =>
        ['present.' + i + '.key', 'present.' + i + '.value']).flat()];
      const mimiOutputs = [
        'present_offset', ...Array.from({ length: DECODER_LAYERS }, (_, i) =>
          ['present.' + i + '.key', 'present.' + i + '.value']).flat(),
        ...Object.keys(CONV_STATES).map(n => n.replace('past_', 'present_')),
      ];
      const flowOpts: Ort.InferenceSession.SessionOptions = {
        executionProviders: [provider], graphOptimizationLevel: 'all',
        ...(gpu ? { preferredOutputLocation: Object.fromEntries(flowOutputs.map(name => [name, 'gpu-buffer'])) } : {}),
        ...(graphs.flowLmData ? {
          externalData: [{ path: graphs.flowLmData.split('/').pop()!, data: this.base + graphs.flowLmData }],
        } : {}),
      };
      const mimiOpts: Ort.InferenceSession.SessionOptions = {
        executionProviders: [provider], graphOptimizationLevel: 'all',
        ...(gpu ? { preferredOutputLocation: Object.fromEntries(mimiOutputs.map(name => [name, 'gpu-buffer'])) } : {}),
        ...(graphs.mimiDecoderData ? {
          externalData: [{ path: graphs.mimiDecoderData.split('/').pop()!, data: this.base + graphs.mimiDecoderData }],
        } : {}),
      };
      this.flowLm = await ort.InferenceSession.create(this.base + graphs.flowLm, flowOpts);
      this.onProgress?.({ phase: 'loading', step: 3, total: 4 });
      checkSignal(context.signal);
      this.mimiDecoder = await ort.InferenceSession.create(this.base + graphs.mimiDecoder, mimiOpts);
      if (!this.flowLm.inputNames.includes('text_ids') ||
        !this.flowLm.outputNames.includes('latent_out') ||
        !this.mimiDecoder.inputNames.includes('latent') ||
        !this.mimiDecoder.outputNames.includes('audio')) {
        throw new Error('Phonon ONNX graph ABI mismatch');
      }
      checkSignal(context.signal);
      this.loadMs = performance.now() - start;
      this.status = 'ready';
      this.onProgress?.({ phase: 'loading', step: 4, total: 4 });
    } catch (e) {
      this.status = 'error';
      try { await this.releaseSessions(); }
      catch (cleanup) { throw Object.assign(new Error('Phonon load and cleanup failed'), { errors: [e, cleanup] }); }
      throw e;
    }
  }

  private async loadVoice(name: string): Promise<Ort.Tensor> {
    if (!this.context || !this.ort) throw new Error('Phonon not loaded');
    if (!/^(default|[A-Za-z0-9_-]{8,30})$/.test(name))
      throw new Error('Phonon voice must be default or a published voice file identifier');
    const raw = await this.context.assets.resolve({
      id: 'phonon-voice-' + name,
      path: 'voices/' + name + '.bin',
    });
    if (raw.byteLength === 0 || raw.byteLength % (4 * VOICE_DIM) !== 0)
      throw new Error('Phonon voice embedding must have [1,512,frames] Float32 data');
    const values = new Float32Array(raw);
    if (!values.every(Number.isFinite)) throw new Error('Phonon voice embedding contains non-finite values');
    return new this.ort.Tensor('float32', values, [1, VOICE_DIM, values.length / VOICE_DIM]);
  }

  private async runFlow(past: State, args: {
    text?: number[];
    voice?: Ort.Tensor;
    latent?: Ort.Tensor;
    noise?: Ort.Tensor;
  }): Promise<Ort.InferenceSession.OnnxValueMapType> {
    const ort = this.ort!, emptyF32 = (dims: number[]) =>
      new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
    const text = args.text ?? [];
    return this.flowLm!.run({
      text_ids: new ort.Tensor('int64', BigInt64Array.from(text, BigInt), [1, text.length]),
      voice_latents: args.voice ?? emptyF32([1, VOICE_DIM, 0]),
      latent: args.latent ?? emptyF32([1, 0, LATENT_DIM]),
      noise: args.noise ?? emptyF32([1, LATENT_DIM]),
      num_speakers: new ort.Tensor('int64', BigInt64Array.of(1n), [1]),
      duration_delta: emptyF32([1]),
      ...past,
    });
  }

  async run(input: PhononInput, config: PhononConfig = {}, signal?: AbortSignal): Promise<AudioInferenceResult> {
    if (this.status !== 'ready' || !this.flowLm || !this.mimiDecoder || !this.tokenizer || !this.ort)
      throw new Error('PhononPipeline not loaded');
    if (this.busy) throw new Error('Phonon inference already running');
    checkSignal(signal);
    const text = normalizeText(input.text);
    const temperature = config.temperature ?? 0.3;
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2)
      throw new Error('Phonon temperature must be between 0 and 2');
    this.busy = true;
    this.status = 'running';
    const start = performance.now();
    const operation = this.synthesize(text, config.voice ?? input.voice ?? 'default', temperature, signal, config.onChunk);
    this.inFlight = operation.then(() => undefined, () => undefined);
    try {
      const samples = await operation;
      const durationSeconds = samples.length / RATE;
      const warnings = checkAudioValid(samples, RATE, 1, durationSeconds);
      return {
        kind: 'audio', samples, sampleRate: RATE, channels: 1, durationSeconds,
        receipt: createInferenceReceipt({
          manifest: this.manifest, backend: this.context!.backend, loadMs: this.loadMs,
          compileMs: 0, inferenceStart: start,
          inputSummary: text.length + ' characters',
          outputSummary: samples.length + ' actual PCM samples at 24000 Hz',
          warnings,
        }),
      };
    } finally {
      this.busy = false;
      if (this.status === 'running') this.status = 'ready';
      this.inFlight = null;
    }
  }

  private async synthesize(text: string, voiceName: string, temperature: number,
    signal?: AbortSignal, onChunk?: (samples: Float32Array) => void): Promise<Float32Array> {
    const ort = this.ort!;
    const voice = await this.loadVoice(voiceName);
    const f32 = (dims: number[], data?: Float32Array) =>
      new ort.Tensor('float32', data ?? new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
    const i64 = (n: number) => new ort.Tensor('int64', BigInt64Array.of(BigInt(n)), [1]);
    const sections = text.split(/(?<=[.!?])\s+/u);
    const result: Float32Array[] = [];
    let total = 0;
    try {
      for (const sentence of sections) {
        checkSignal(signal);
        const tokens = this.tokenizer!.encode(sentence, { add_special_tokens: false });
        if (!tokens.length) throw new Error('Phonon tokenizer produced no tokens');
        if (tokens.length > 50) throw new Error('Phonon sentence exceeds 50 tokens; split it into shorter sentences');
        const prompted = await this.runFlow(cache(ort, FLOW_LAYERS), { voice });
        const voiceState = stateFrom(prompted);
        let state: State = {};
        let decoder: State = {};
        let latent: Ort.Tensor = f32([1, 1, LATENT_DIM],
          new Float32Array(LATENT_DIM).fill(NaN)); // published start-of-speech sentinel
        try {
          const withText = await this.runFlow(voiceState, { text: tokens });
          state = stateFrom(withText);
          decoder = { past_offset: i64(0), ...cache(ort, DECODER_LAYERS) };
          for (const [key, dims] of Object.entries(CONV_STATES)) decoder[key] = f32([...dims]);
          let countdown: number | null = null;
          const maxFrames = Math.ceil((tokens.length / 3 + 2) * FRAME_RATE);
          for (let frame = 0; frame < maxFrames; frame++) {
            checkSignal(signal);
            const noise = f32([1, LATENT_DIM], Float32Array.from({ length: LATENT_DIM },
              () => temperature === 0 ? 0 : randNormal() * Math.sqrt(temperature)));
            const flow = await this.runFlow(state, { latent, noise });
            const newState = stateFrom(flow);
            const newLatent = requireOutput(flow, 'latent_out');
            const eos = requireOutput(flow, 'eos_logit');
            releaseGpu(state, { latent });
            state = newState;
            latent = newLatent;
            const pcm = await this.mimiDecoder!.run({ latent, ...decoder });
            const nextDecoder = stateFrom(pcm);
            releaseGpu(decoder);
            decoder = nextDecoder;
            const audio = requireOutput(pcm, 'audio');
            if (!(audio.data instanceof Float32Array) || audio.data.length !== 1920 ||
              !audio.data.every(Number.isFinite)) {
              throw new Error('Phonon decoder produced malformed PCM; refusing silent placeholder');
            }
            const chunk = new Float32Array(audio.data);
            result.push(chunk);
            total += chunk.length;
            onChunk?.(chunk);
            if (countdown === null && Number(eos.data[0]) > EOS_THRESHOLD)
              countdown = sentence.split(' ').length <= 4 ? 3 : 1;
            if (countdown !== null && countdown-- === 0) break;
          }
        } finally {
          releaseGpu(state, decoder, { latent }, voiceState);
        }
      }
    } finally { releaseGpu({ voice }); }
    if (!total) throw new Error('Phonon decoder generated no PCM');
    const samples = new Float32Array(total);
    let offset = 0;
    for (const chunk of result) { samples.set(chunk, offset); offset += chunk.length; }
    if (!samples.some(x => Math.abs(x) > 1e-6))
      throw new Error('Phonon inference produced only silence');
    return samples;
  }

  private async releaseSessions(): Promise<void> {
    const flow = this.flowLm, mimi = this.mimiDecoder;
    this.flowLm = this.mimiDecoder = null;
    this.ort = null; this.tokenizer = null;
    const settled = await Promise.allSettled([flow?.release(), mimi?.release()]);
    const failures = settled.filter(s => s.status === 'rejected');
    if (failures.length) throw new Error('Phonon ONNX session release failed');
  }
  dispose(): Promise<void> {
    if (this.disposing) return this.disposing;
    this.status = 'disposed';
    this.disposing = (async () => {
      await this.inFlight;
      await this.releaseSessions();
      this.context = null;
    })();
    return this.disposing;
  }
}
