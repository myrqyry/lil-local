import { CompiledModel, Tensor } from '@litertjs/core'
import { traceArray, traceTensor, type GeneratorTraceEvent } from './generator-trace'

export interface TalkerConfig {
  /** Number of KV cache slots (64 for talker_fp32) */
  numKvSlots: number
  /** Hidden dimension (1024 for Qwen3-TTS) */
  hiddenDim: number
  /** Codec vocabulary size (3072) */
  codecVocab: number
  cacheLen: number
  kvNames: string[]
  kvShapes: number[][]
  /** Accelerator to move input tensors to before running (matches reference) */
  accelerator?: 'wasm' | 'webgpu'
  onTrace?: (event: GeneratorTraceEvent) => void
}

const DEFAULT_CONFIG: TalkerConfig = {
  numKvSlots: 64,
  hiddenDim: 1024,
  codecVocab: 3072,
  cacheLen: 32,
  kvNames: [],
  kvShapes: [],
  accelerator: 'wasm',
}

async function toInputTensor(data: Float32Array | Int32Array, shape: number[], accelerator: 'wasm' | 'webgpu'): Promise<Tensor> {
  return new Tensor(data, shape).moveTo(accelerator)
}

function readFloat32(tensor: Tensor): Float32Array {
  const arr = tensor.toTypedArray()
  // Cast guards against newer TS libs whose TypedArray union includes
  // non-ArrayLike<number> members (BigInt64Array/Float16Array); these tensors are float32.
  return arr instanceof Float32Array ? arr : Float32Array.from(arr as ArrayLike<number>)
}

export class Talker {
  private config: TalkerConfig

  constructor(
    private model: CompiledModel,
    config?: Partial<TalkerConfig>,
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  get numKvSlots(): number { return this.config.numKvSlots }

  /** KV cache lives in the JS heap (Float32Array[]) so WASM managed
   * buffers are never held across calls — matches the PodQast reference
   * and avoids renderer WASM memory exhaustion during decode. */
  createEmptyKv(): Float32Array[] {
    const kv: Float32Array[] = []
    for (let i = 0; i < this.config.kvNames.length; i++) {
      kv.push(new Float32Array(this.config.kvShapes[i].reduce((a, b) => a * b, 1)))
    }
    return kv
  }

  async prefill(
    embeddings: Float32Array,
    kvCache: Float32Array[],
    seqLen?: number,
  ): Promise<{ kvCache: Float32Array[] }> {
    this.trace({
      stage: 'talker-prefill',
      phase: 'start',
      tensors: [
        traceArray('embeddings', 'float32', [1, 32, this.config.hiddenDim]),
        traceArray('input_pos', 'int32', [32]),
        traceArray('mask', 'float32', [1, 1, 32, this.config.cacheLen]),
        ...this.config.kvNames.map((name, index) =>
          traceArray(name, 'float32', this.config.kvShapes[index])),
      ],
    })
    const inputs: Record<string, Promise<Tensor>> = {}
    for (let i = 0; i < this.config.kvNames.length; i++) {
      inputs[this.config.kvNames[i]] = toInputTensor(kvCache[i], this.config.kvShapes[i], this.config.accelerator!)
    }
    const sl = seqLen ?? (embeddings.length / this.config.hiddenDim) | 0
    const negInf = -1e9
    const mask = new Float32Array(1 * 1 * 32 * this.config.cacheLen).fill(negInf)
    for (let i = 0; i < 32 && i < sl; i++) {
      for (let j = 0; j <= i && j < sl; j++) {
        mask[i * this.config.cacheLen + j] = 0
      }
    }

    const inputPos = new Int32Array(32)
    for (let i = 0; i < 32; i++) inputPos[i] = i

    const paddedEmb = new Float32Array(32 * this.config.hiddenDim)
    paddedEmb.set(embeddings)

    inputs['embeddings'] = toInputTensor(paddedEmb, [1, 32, this.config.hiddenDim], this.config.accelerator!)
    inputs['input_pos'] = toInputTensor(inputPos, [32], this.config.accelerator!)
    inputs['mask'] = toInputTensor(mask, [1, 1, 32, this.config.cacheLen], this.config.accelerator!)

    const resolved: Record<string, Tensor> = {}
    for (const [key, promise] of Object.entries(inputs)) resolved[key] = await promise
    const startedAt = performance.now()
    const result = await this.model.run('prefill_32', resolved)

    const outKv: Float32Array[] = []
    for (let i = 0; i < this.config.kvNames.length; i++) {
      outKv.push(readFloat32(result[this.config.kvNames[i]] as Tensor))
    }

    this.trace({ stage: 'talker-prefill', phase: 'end', durationMs: performance.now() - startedAt })
    this.trace({
      stage: 'talker-output-read',
      tensors: this.config.kvNames.map((name) => traceTensor(name, result[name] as Tensor)),
    })

    return { kvCache: outKv }
  }

  async decode(
    embeddings: Float32Array,
    kvCache: Float32Array[],
    pos?: number,
  ): Promise<{ logits: Float32Array; hidden: Float32Array; kvCache: Float32Array[] }> {
    const inputs: Record<string, Promise<Tensor>> = {}
    for (let i = 0; i < this.config.kvNames.length; i++) {
      inputs[this.config.kvNames[i]] = toInputTensor(kvCache[i], this.config.kvShapes[i], this.config.accelerator!)
    }
    const p = pos ?? 0
    const negInf = -1e9
    const mask = new Float32Array(1 * 1 * 1 * this.config.cacheLen).fill(negInf)
    for (let i = 0; i <= p && i < this.config.cacheLen; i++) mask[i] = 0

    inputs['embeddings'] = toInputTensor(embeddings, [1, 1, this.config.hiddenDim], this.config.accelerator!)
    inputs['input_pos'] = toInputTensor(new Int32Array([p]), [1], this.config.accelerator!)
    inputs['mask'] = toInputTensor(mask, [1, 1, 1, this.config.cacheLen], this.config.accelerator!)

    const resolved: Record<string, Tensor> = {}
    for (const [key, promise] of Object.entries(inputs)) resolved[key] = await promise
    const result = await this.model.run('decode', resolved)
    const logits = readFloat32(result['logits'] as Tensor)
    const cb0Logits = logits.slice(0, this.config.codecVocab)
    const hidden = logits.slice(this.config.codecVocab)

    const outKv: Float32Array[] = []
    for (let i = 0; i < this.config.kvNames.length; i++) {
      outKv.push(readFloat32(result[this.config.kvNames[i]] as Tensor))
    }
    this.trace({
      stage: 'talker-output-read',
      tensors: [
        traceTensor('logits', result.logits as Tensor),
        ...this.config.kvNames.map((name) => traceTensor(name, result[name] as Tensor)),
      ],
    })

    return { logits: cb0Logits, hidden, kvCache: outKv }
  }

  private trace(event: GeneratorTraceEvent): void {
    this.config.onTrace?.(event)
  }
}
