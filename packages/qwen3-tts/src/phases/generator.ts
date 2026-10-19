import type { RuntimeContext, PipelineProgress, ModelManifest } from '@lil-local/inference-core';
import { InferenceError } from '@lil-local/inference-core';
import { BPETokenizer } from '../tokenizer';
import { Talker } from '../talker';
import { MTP } from '../mtp';
import { sample, type SampleOpts } from '../sampler';
import { parseNpy, parseNpz } from '../npy-parser';
import { buildPrompt } from '../prompt';
import { createQwen3TtsManifest, type Qwen3TtsVariant } from '../manifest';
import { discoverMtpShapes, discoverTalkerShapes } from '../shape-discovery';
import { parseFp16Npy, type Fp16Table } from '../fp16-table';
import { HIDDEN, CODEC_VOCAB, CODEC_EOS, NEG_INF, LANGUAGE_IDS, DEFAULTS, type QwenTtsInput, type QwenTtsConfig } from '../types';
import { packCodecFrames, type CodecFrames } from '../codec-frames';
import { traceArray, type GeneratorTraceEvent } from '../generator-trace';

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export interface GeneratorPhaseOptions {
  onProgress?: (progress: PipelineProgress) => void;
  onTrace?: (event: GeneratorTraceEvent) => void;
}

export class GeneratorPhase {
  readonly manifest: ModelManifest;
  readonly name = 'generator';
  loadMs = 0;
  compileMs = 0;
  inferenceMs = 0;

  private readonly variant: Qwen3TtsVariant;
  private readonly onProgress?: (progress: PipelineProgress) => void;
  private readonly onTrace?: (event: GeneratorTraceEvent) => void;
  private context?: RuntimeContext;
  private tokenizer?: BPETokenizer;
  private talker?: Talker;
  private mtp?: MTP;
  private codecEmb?: Float32Array;
  private mtpEmb?: Float32Array;
  private textEmbData?: Fp16Table;
  private projW1?: Float32Array;
  private projB1?: Float32Array;
  private projW2?: Float32Array;
  private projB2?: Float32Array;

  constructor(variant: Qwen3TtsVariant, options: GeneratorPhaseOptions = {}) {
    this.variant = variant;
    this.onProgress = options.onProgress;
    this.onTrace = options.onTrace;
    this.manifest = createQwen3TtsManifest(variant);
  }

  async load(context: RuntimeContext): Promise<void> {
    this.context = context;
    const loadStart = performance.now();
    this.report({ phase: 'loading', step: 0, total: 7 });
    const tokData = await context.assets.resolve({ id: 'tokenizer', path: 'tokenizer.json' });
    this.tokenizer = new BPETokenizer(JSON.parse(new TextDecoder().decode(tokData)));
    this.report({ phase: 'loading', step: 1, total: 7 });
    this.codecEmb = await context.liteRt.loadNpy('tables/codec_embedding_fp32.npy');
    this.report({ phase: 'loading', step: 2, total: 7 });
    this.mtpEmb = await context.liteRt.loadNpy('tables/mtp_embeddings_fp16.npy');
    this.report({ phase: 'loading', step: 3, total: 7 });
    this.textEmbData = parseFp16Npy(await context.liteRt.fetchBuffer('tables/text_embedding_fp16.npy'));
    this.report({ phase: 'loading', step: 4, total: 7 });
    const projBuf = await context.assets.resolve({ id: 'text-projection', path: 'tables/text_projection_fp32.npz' });
    const proj = await parseNpz(projBuf);
    this.projW1 = proj['w1'];
    this.projB1 = proj['b1'];
    this.projW2 = proj['w2'];
    this.projB2 = proj['b2'];
    this.report({ phase: 'loading', step: 5, total: 7 });
    const compileStart = performance.now();
    this.trace({ stage: 'talker-compile', phase: 'start' });
    const talkerModel = await context.liteRt.loadModel(this.variant.talker);
    this.trace({ stage: 'talker-compile', phase: 'end' });
    this.report({ phase: 'loading', step: 6, total: 7 });
    this.trace({ stage: 'mtp-compile', phase: 'start' });
    const mtpModel = await context.liteRt.loadModel(this.variant.mtp);
    this.trace({ stage: 'mtp-compile', phase: 'end' });
    this.compileMs = performance.now() - compileStart;
    const talkerShapes = discoverTalkerShapes(talkerModel);
    const mtpShapes = discoverMtpShapes(mtpModel);
    const accelerator = this.context!.backend === 'webgpu' ? 'webgpu' : 'wasm'
    this.talker = new Talker(talkerModel, { ...talkerShapes, accelerator, onTrace: (event) => this.trace(event) });
    this.mtp = new MTP(mtpModel, {
      mtpEmbeddings: this.mtpEmb,
      codecEmbeddings: this.codecEmb,
      numCacheSlots: mtpShapes.cacheLen,
      cacheShape: mtpShapes.kvShape,
      accelerator,
      onTrace: (event) => this.trace(event),
    });
    this.loadMs = performance.now() - loadStart;
  }

  async generate(input: QwenTtsInput, config: QwenTtsConfig, signal?: AbortSignal): Promise<CodecFrames> {
    const ctx = this.context!;
    const cfg = { ...DEFAULTS, ...config };
    const lang = LANGUAGE_IDS[cfg.language] ?? LANGUAGE_IDS.english;
    const inferenceStart = performance.now();
    try {
      const voicePath = `voices/${cfg.voice}.npy`;
      const speakerBuf = await ctx.assets.resolve({ id: 'voice', path: voicePath, optional: true }, { signal });
      const speakerEmb = parseNpy(speakerBuf);
      const { prefill, trailing, ttsPad } = buildPrompt(
        input.text,
        speakerEmb,
        lang,
        this.tokenizer!,
        this.codecEmb!,
        this.textEmbData!,
        (row) => this.projectText(row),
      );
      if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled before prefill');
      this.report({ phase: 'prefill', step: 0, total: 1 });
      const kv = this.talker!.createEmptyKv();
      const sl = prefill.length / HIDDEN;
      const { kvCache } = await this.talker!.prefill(prefill, kv, sl);
      // prefill_32 has no logits output (reference flow): the first decode
      // call, seeded with the last prefill embedding row, produces logits.
      let pos = sl - 1;
      const lastRow = prefill.slice((sl - 1) * HIDDEN, sl * HIDDEN);
      let { logits, hidden, kvCache: currentKv } = await this.talker!.decode(lastRow, kvCache, pos);
      const sampleOpts: SampleOpts = {
        temperature: cfg.temperature,
        topK: cfg.topK,
        repetitionPenalty: cfg.repetitionPenalty,
        prevTokens: [],
      };
      const allFrames: number[][] = [];
      let currentLogits = logits;
      let currentHidden = hidden;
      const maxFrames = cfg.maxFrames;
      for (let frame = 0; frame < maxFrames; frame++) {
        if (signal?.aborted) throw new InferenceError('CANCELLED', 'Cancelled during generation');
        this.report({ phase: 'decode', step: frame, total: maxFrames });
        const scores = new Float32Array(currentLogits);
        for (let i = 2048; i < CODEC_VOCAB; i++) scores[i] = NEG_INF;
        scores[CODEC_EOS] = 0;
        if (frame < 2) scores[CODEC_EOS] = NEG_INF;
        for (const token of sampleOpts.prevTokens) {
          scores[token] = scores[token] > 0 ? scores[token] / sampleOpts.repetitionPenalty : scores[token] * sampleOpts.repetitionPenalty;
        }
        const cb0 = sample(scores, { ...sampleOpts, prevTokens: [] });
        if (cb0 === CODEC_EOS) break;
        sampleOpts.prevTokens.push(cb0);
        this.report({ phase: 'mtp', step: frame, total: maxFrames });
        const residual = await this.mtp!.predict(currentHidden, cb0, { temperature: cfg.temperature, topK: cfg.topK });
        allFrames.push([cb0, ...residual]);
        const frameIdx = frame < trailing.length ? frame : trailing.length - 1;
        const textCond = frameIdx >= 0 ? trailing[frameIdx] : ttsPad;
        const sumEmb = new Float32Array(HIDDEN);
        const cb0Emb = this.codecEmb!.slice(cb0 * HIDDEN, (cb0 + 1) * HIDDEN);
        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += cb0Emb[i];
        for (let r = 0; r < residual.length; r++) {
          const re = this.mtpEmb!.slice(r * CODEC_VOCAB * HIDDEN + residual[r] * HIDDEN, r * CODEC_VOCAB * HIDDEN + (residual[r] + 1) * HIDDEN);
          for (let i = 0; i < HIDDEN; i++) sumEmb[i] += re[i] / residual.length;
        }
        for (let i = 0; i < HIDDEN; i++) sumEmb[i] += textCond[i];
        pos += 1;
        const result = await this.talker!.decode(sumEmb, currentKv, pos);
        currentLogits = result.logits;
        currentHidden = result.hidden;
        currentKv = result.kvCache;
        this.trace({
          stage: 'state-update',
          frame,
          tensors: [
            traceArray('logits', 'float32', [1, CODEC_VOCAB]),
            traceArray('hidden', 'float32', [1, HIDDEN]),
          ],
        });
      }
      this.inferenceMs = performance.now() - inferenceStart;
      return packCodecFrames(allFrames);
    } catch (e) {
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e });
    }
  }

  dispose(): void {
    this.talker = undefined;
    this.mtp = undefined;
    this.tokenizer = undefined;
    this.codecEmb = undefined;
    this.mtpEmb = undefined;
    this.textEmbData = undefined;
    this.projW1 = undefined;
    this.projB1 = undefined;
    this.projW2 = undefined;
    this.projB2 = undefined;
    this.context = undefined;
  }

  private projectText(row: Float32Array): Float32Array {
    const hiddenDim = HIDDEN * 4;
    const inputDim = this.projW1!.length / hiddenDim;
    const h = new Float32Array(hiddenDim);
    for (let i = 0; i < hiddenDim; i++) {
      let acc = this.projB1![i];
      for (let j = 0; j < inputDim; j++) acc += this.projW1![j * hiddenDim + i] * row[j];
      h[i] = silu(acc);
    }
    const out = new Float32Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      let acc = this.projB2![i];
      for (let j = 0; j < hiddenDim; j++) acc += this.projW2![j * HIDDEN + i] * h[j];
      out[i] = acc;
    }
    return out;
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }

  private trace(event: GeneratorTraceEvent): void {
    this.onTrace?.(event);
  }

}
