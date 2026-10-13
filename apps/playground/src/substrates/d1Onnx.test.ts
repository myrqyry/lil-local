import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@litert-playground/inference-core'
import { D1OnnxWebGpu } from '@litert-playground/decision-onnx'
import { d1OnnxSubstrate } from './d1Onnx'

vi.mock('@litert-playground/decision-onnx', () => ({
  D1OnnxWebGpu: { load: vi.fn() },
}))

describe('ONNX Runtime Web substrate integration', () => {
  it('uses the real D1 entrypoint lazily, passing inference and cleanup through the host', async () => {
    const decide = vi.fn(async () => ({
      type: 'noul' as const, label: 'true',
      probabilities: { false: 0.1, true: 0.9 }, rawLogits: [0, 1],
      elapsedMs: 5, provider: 'onnxruntime-web' as const,
      backend: 'webgpu' as const, modelRevision: 'test', calibrated: false as const,
    }))
    const dispose = vi.fn(async () => undefined)
    vi.mocked(D1OnnxWebGpu.load).mockResolvedValueOnce({
      decide, dispose,
    } as unknown as D1OnnxWebGpu)
    const runner = new BrowserSubstrateRunner(d1OnnxSubstrate)
    expect(d1OnnxSubstrate.requestedBackends).toEqual(['webgpu'])
    expect(d1OnnxSubstrate.format).toBe('onnx')
    expect(D1OnnxWebGpu.load).not.toHaveBeenCalled()
    const phase = vi.fn()
    await runner.load(phase)
    const question = { type: 'noul', instructions: 'Is this a greeting?' } as const
    const result = await runner.run({ state: 'hello', question })
    expect(decide).toHaveBeenCalledExactlyOnceWith('hello', question)
    expect(result.provider).toBe('onnxruntime-web')
    expect(result.backend).toBe('webgpu')
    await runner.dispose()
    expect(dispose).toHaveBeenCalledOnce()
  })
})
