import {
  type PipelineStatus,
  type RuntimeContext,
  type AudioInferenceResult,
  type PipelineProgress,
  InferenceError,
  checkAudioValid,
  createInferenceReceipt,
} from '@lil-local/inference-core'
import { createQwen3TtsManifest, qwen3TtsVariants, type Qwen3TtsVariant } from './manifest'
import { DEFAULTS, type QwenTtsInput, type QwenTtsConfig } from './types'
import { GeneratorPhase } from './phases/generator'
import { DecoderPhase } from './phases/decoder'
import { runHostGenerator, runHostDecoder } from './workers/host'

export type { QwenTtsInput, QwenTtsConfig }

export interface Qwen3TtsPipelineOptions {
  workerBase?: string
  modelBase?: string
  executionMode?: 'auto' | 'direct' | 'workers'
}

const WORKER_BASE = '/litert-wasm/'
const MODEL_BASE = '/models/qwen3-tts/'

export class Qwen3TtsPipeline {
  readonly manifest
  status: PipelineStatus = 'idle'

  onProgress?: (progress: PipelineProgress) => void

  private readonly variant: Qwen3TtsVariant
  private readonly workerBase: string
  private readonly modelBase: string
  private readonly executionMode: NonNullable<Qwen3TtsPipelineOptions['executionMode']>
  private context: RuntimeContext | null = null
  private disposed = false

  constructor(variant: Qwen3TtsVariant = qwen3TtsVariants.fp32, options: Qwen3TtsPipelineOptions = {}) {
    this.variant = variant
    this.workerBase = options.workerBase ?? WORKER_BASE
    this.modelBase = options.modelBase ?? MODEL_BASE
    this.executionMode = options.executionMode ?? 'auto'
    this.manifest = createQwen3TtsManifest(variant)
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.disposed) throw new InferenceError('INFERENCE_FAILED', 'Pipeline disposed')
    if (this.status === 'ready') return

    this.context = context
    this.status = 'loading'
    this.report({ phase: 'loading', step: 0, total: 1 })

    if (!context.liteRt || !context.assets) {
      this.status = 'error'
      throw new InferenceError('INFERENCE_FAILED', 'Invalid runtime context')
    }

    this.status = 'ready'
  }

  async run(input: QwenTtsInput, config?: QwenTtsConfig, signal?: AbortSignal): Promise<AudioInferenceResult> {
    if (this.status !== 'ready') throw new InferenceError('INFERENCE_FAILED', 'Pipeline not ready')

    this.status = 'running'
    const cfg = { ...DEFAULTS, ...config }
    const inferenceStart = performance.now()
    const backend = this.context?.backend ?? 'wasm'

    try {
      let audio: Float32Array
      let phases

      const useDirect = this.executionMode === 'direct' || (this.executionMode === 'auto' && typeof Worker === 'undefined')

      if (useDirect) {
        const genPhase = new GeneratorPhase(this.variant, { onProgress: (p) => this.report(p) })
        let frames
        try {
          await genPhase.load(this.context!)
          frames = await genPhase.generate(input, cfg, signal)
        } finally {
          genPhase.dispose()
        }

        const decPhase = new DecoderPhase(this.variant, { onProgress: (p) => this.report(p) })
        try {
          await decPhase.load(this.context!)
          audio = await decPhase.decode(frames, signal)
        } finally {
          decPhase.dispose()
        }

        phases = [
          {
            name: genPhase.name,
            backend,
            loadMs: genPhase.loadMs,
            compileMs: genPhase.compileMs,
            inferenceMs: genPhase.inferenceMs,
          },
          {
            name: decPhase.name,
            backend,
            loadMs: decPhase.loadMs,
            compileMs: decPhase.compileMs,
            inferenceMs: decPhase.inferenceMs,
          },
        ]
      } else {
        const genWorker = new Worker(this.workerBase + 'generator-worker.js')
        let gen: Awaited<ReturnType<typeof runHostGenerator>>
        try {
          gen = await runHostGenerator(genWorker, this.variant, this.modelBase, input, cfg, (p) => this.report({ ...p }))
        } finally {
          genWorker.terminate()
        }

        const decWorker = new Worker(this.workerBase + 'decoder-worker.js')
        let dec: Awaited<ReturnType<typeof runHostDecoder>>
        try {
          dec = await runHostDecoder(decWorker, this.variant, this.modelBase, gen.frames, (p) => this.report({ ...p }))
        } finally {
          decWorker.terminate()
        }

        audio = dec.audio
        phases = [gen.phaseReceipt, dec.phaseReceipt]
      }

      const duration = audio.length / 24000
      const warnings = checkAudioValid(audio, 24000, 1, duration)

      if (warnings.length > 0) {
        console.warn('Qwen3TTS output warnings:', warnings)
      }

      this.status = 'ready'
      return {
        kind: 'audio',
        samples: audio,
        sampleRate: 24000,
        channels: 1,
        durationSeconds: duration,
        receipt: createInferenceReceipt({
          manifest: this.manifest,
          backend,
          loadMs: 0,
          compileMs: 0,
          inferenceStart,
          inputSummary: `${input.text.length} characters`,
          outputSummary: `${audio.length} samples at 24000Hz, 1 channel`,
          warnings,
          phases,
        }),
      }
    } catch (e) {
      this.status = 'ready'
      throw e instanceof InferenceError ? e : new InferenceError('INFERENCE_FAILED', String(e), { cause: e })
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    this.context = null
    this.status = 'disposed'
  }

  // ---- internals ----

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress)
  }
}
