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
  const promise = new Promise<T>((res) => { resolve = res })
  return { promise, resolve }
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
