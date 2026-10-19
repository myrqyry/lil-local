import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AutoTokenizer } from '@huggingface/transformers'
import * as ort from 'onnxruntime-web/webgpu'
import { D1OnnxWebGpu } from './runtime'

vi.mock('@huggingface/transformers', () => ({
  AutoTokenizer: { from_pretrained: vi.fn() },
}))
vi.mock('onnxruntime-web/webgpu', () => ({
  InferenceSession: { create: vi.fn() },
}))

const inputs = ['input_ids', 'attention_mask', 'prefix_len', 'marker_pos', 'marker_mask', 'qtype']

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

describe('D1 ONNX cooperative load cancellation', () => {
  beforeEach(() => {
    vi.stubGlobal('navigator', { gpu: {} })
    vi.mocked(AutoTokenizer.from_pretrained).mockReset()
    vi.mocked(ort.InferenceSession.create).mockReset()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('avoids loading dependencies for already aborted requests', async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(D1OnnxWebGpu.load(undefined, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(AutoTokenizer.from_pretrained).not.toHaveBeenCalled()
    expect(ort.InferenceSession.create).not.toHaveBeenCalled()
  })

  it('aborts after tokenizer completes before initiating a graph load', async () => {
    const controller = new AbortController()
    const tokenizer = deferred<unknown>()
    vi.mocked(AutoTokenizer.from_pretrained).mockReturnValueOnce(tokenizer.promise as never)
    const loading = D1OnnxWebGpu.load(undefined, controller.signal)
    // Allow the dynamic-import microtasks to reach the tokenizer.
    await vi.waitFor(() => expect(AutoTokenizer.from_pretrained).toHaveBeenCalledOnce())
    controller.abort()
    tokenizer.resolve({})
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(ort.InferenceSession.create).not.toHaveBeenCalled()
  })

  it('holds all concurrent dispose callers until the same runtime release completes', async () => {
    const gate = deferred<void>()
    const release = vi.fn(() => gate.promise)
    vi.mocked(AutoTokenizer.from_pretrained).mockResolvedValueOnce({} as never)
    vi.mocked(ort.InferenceSession.create).mockResolvedValueOnce({
      inputNames: inputs, outputNames: ['logits'], release,
    } as never)

    const instance = await D1OnnxWebGpu.load()
    const first = instance.dispose()
    const second = instance.dispose()
    let firstSettled = false
    let secondSettled = false
    void first.then(() => { firstSettled = true })
    void second.then(() => { secondSettled = true })
    await Promise.resolve()
    await Promise.resolve()
    expect(firstSettled).toBe(false)
    expect(secondSettled).toBe(false)
    gate.resolve()
    await Promise.all([first, second])
    expect(release).toHaveBeenCalledTimes(1)
  })

  it('reports the same release failure to every disposal caller, including later calls', async () => {
    const gate = deferred<void>()
    const release = vi.fn(() => gate.promise)
    vi.mocked(AutoTokenizer.from_pretrained).mockResolvedValueOnce({} as never)
    vi.mocked(ort.InferenceSession.create).mockResolvedValueOnce({
      inputNames: inputs, outputNames: ['logits'], release,
    } as never)

    const instance = await D1OnnxWebGpu.load()
    const first = instance.dispose()
    const second = instance.dispose()
    const observed = Promise.allSettled([first, second])
    const releaseError = new Error('GPU memory release failed')
    gate.reject(releaseError)
    const outcomes = await observed
    expect(outcomes).toEqual([
      { status: 'rejected', reason: releaseError },
      { status: 'rejected', reason: releaseError },
    ])
    await expect(instance.dispose()).rejects.toBe(releaseError)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it('keeps both the primary model-ABI failure and the release failure visible', async () => {
    const releaseError = new Error('GPU release broke')
    const release = vi.fn(async () => { throw releaseError })
    vi.mocked(AutoTokenizer.from_pretrained).mockResolvedValueOnce({} as never)
    vi.mocked(ort.InferenceSession.create).mockResolvedValueOnce({
      inputNames: ['input_ids'],
      outputNames: ['logits'],
      release,
    } as never)

    try {
      await D1OnnxWebGpu.load()
      throw new Error('Expected the model load to fail')
    } catch (error) {
      expect(error).toMatchObject({ name: 'D1LoadCleanupError' })
      const failures = (error as { errors: unknown[] }).errors
      expect(failures).toHaveLength(2)
      expect(String(failures[0])).toContain('Unexpected D1 ONNX input/output ABI')
      expect(failures[1]).toBe(releaseError)
    }
    expect(release).toHaveBeenCalledOnce()
  })

  it('releases a graph that finishes compiling after cancellation without signaling ready', async () => {
    const controller = new AbortController()
    const session = deferred<unknown>()
    const release = vi.fn(async () => undefined)
    const phases: string[] = []
    vi.mocked(AutoTokenizer.from_pretrained).mockResolvedValueOnce({} as never)
    vi.mocked(ort.InferenceSession.create).mockReturnValueOnce(session.promise as never)

    const loading = D1OnnxWebGpu.load(phase => { phases.push(phase) }, controller.signal)
    await vi.waitFor(() => expect(ort.InferenceSession.create).toHaveBeenCalledOnce())
    controller.abort()
    session.resolve({ inputNames: inputs, outputNames: ['logits'], release })
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(release).toHaveBeenCalledOnce()
    expect(phases).toEqual(['tokenizer', 'graph'])
  })
})
