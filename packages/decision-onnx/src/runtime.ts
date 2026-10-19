/**
 * Explicit browser WebGPU D1 Omni text-decision experiment.
 * Separate from Lil Tart's LiteRT runtime: different graph format/ABI.
 */
import type { PreTrainedTokenizer } from '@huggingface/transformers'
import type * as Ort from 'onnxruntime-web'
import {
  decodeD1Scores,
  encodeD1Text,
  type D1Decision,
  type D1Question,
  type D1Tokenizer,
} from './contract'

export const D1_MODEL_ID = 'onnx-community/d1-omni-600M-ONNX'
export const D1_MODEL_REVISION = '4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0'
const BASE = 'https://huggingface.co/' + D1_MODEL_ID + '/resolve/' + D1_MODEL_REVISION
const GRAPH = BASE + '/onnx/model_quantized.onnx'
const WEIGHTS = BASE + '/onnx/model_quantized.onnx_data'
const EXPECTED_INPUTS = [
  'input_ids', 'attention_mask', 'prefix_len', 'marker_pos', 'marker_mask', 'qtype',
]

export type D1LoadPhase = 'tokenizer' | 'graph' | 'ready'

function checkLoadSignal(signal?: AbortSignal): void {
  if (!signal?.aborted) return
  const error = new Error('D1 ONNX load aborted')
  error.name = 'AbortError'
  throw error
}

/**
 * Download and load only after an explicit user gesture. No model is downloaded
 * by module import, render, discovery, or preflight. No implicit WASM fallback.
 */
export class D1OnnxWebGpu {
  private session: Ort.InferenceSession | null = null
  private tokenizer: D1Tokenizer | null = null
  private ort: typeof Ort | null = null
  private disposed = false
  private busy = false
  private whenIdle: Array<() => void> = []

  static async load(onPhase?: (phase: D1LoadPhase) => void, signal?: AbortSignal): Promise<D1OnnxWebGpu> {
    checkLoadSignal(signal)
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('WebGPU is unavailable in this browser; D1 ONNX load refused (no implicit CPU fallback)')
    }
    const instance = new D1OnnxWebGpu()
    try {
      onPhase?.('tokenizer')
      checkLoadSignal(signal)
      const [{ AutoTokenizer }, ort] = await Promise.all([
        import('@huggingface/transformers'),
        import('onnxruntime-web/webgpu'),
      ])
      checkLoadSignal(signal)
      const tokenizer = await AutoTokenizer.from_pretrained(D1_MODEL_ID, {
        revision: D1_MODEL_REVISION,
      })
      checkLoadSignal(signal)
      instance.tokenizer = tokenizer as unknown as D1Tokenizer
      instance.ort = ort
      onPhase?.('graph')
      checkLoadSignal(signal)
      const session = await ort.InferenceSession.create(GRAPH, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        externalData: [{ path: 'model_quantized.onnx_data', data: WEIGHTS }],
      })
      instance.session = session
      checkLoadSignal(signal)
      if (EXPECTED_INPUTS.some(name => !session.inputNames.includes(name)) ||
          !session.outputNames.includes('logits')) {
        throw new Error('Unexpected D1 ONNX input/output ABI; model update requires review')
      }
      onPhase?.('ready')
      checkLoadSignal(signal)
      return instance
    } catch (error) {
      await instance.dispose()
      throw error
    }
  }

  async decide(state: string, question: D1Question): Promise<D1Decision> {
    if (this.disposed || !this.session || !this.ort || !this.tokenizer) {
      throw new Error('D1 ONNX model is not loaded')
    }
    if (this.busy) throw new Error('D1 inference already running')
    this.busy = true
    const start = performance.now()
    try {
      const { ids, markerPositions, qtype, optionLabels } =
        encodeD1Text(this.tokenizer, state, question)
      const int64 = (data: bigint[], shape: number[]) =>
        new this.ort!.Tensor('int64', BigInt64Array.from(data), shape)
      const feeds = {
        input_ids: int64(ids, [1, ids.length]),
        attention_mask: int64(ids.map(() => 1n), [1, ids.length]),
        prefix_len: int64([0n], [1]),
        marker_pos: int64(markerPositions, [1, markerPositions.length]),
        marker_mask: int64(markerPositions.map(() => 1n), [1, markerPositions.length]),
        qtype: int64([qtype], [1]),
      }
      const outputs = await this.session.run(feeds)
      const logits = outputs.logits
      if (!logits || logits.dims[0] !== 1 || logits.dims[1] !== markerPositions.length) {
        throw new Error('D1 returned an unexpected logits tensor shape')
      }
      const rawLogits = Array.from(logits.data, Number)
      const scored = decodeD1Scores(rawLogits, optionLabels, question.type)
      return {
        type: question.type,
        ...scored,
        rawLogits,
        elapsedMs: Math.round((performance.now() - start) * 100) / 100,
        provider: 'onnxruntime-web',
        backend: 'webgpu',
        modelRevision: D1_MODEL_REVISION,
        calibrated: false,
      }
    } finally {
      this.busy = false
      for (const notify of this.whenIdle.splice(0)) notify()
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    // A navigation/unmount may occur while inference is running. Never release
    // ORT buffers out from under an active session.run().
    if (this.busy) {
      await new Promise<void>(resolve => this.whenIdle.push(resolve))
    }
    const session = this.session
    this.session = null
    this.tokenizer = null
    this.ort = null
    if (session) await session.release()
  }
}
