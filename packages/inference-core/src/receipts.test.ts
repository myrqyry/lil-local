import { describe, expect, it, vi } from 'vitest'
import { createInferenceReceipt } from './receipts'

describe('inference receipts', () => {
  it('adds shared timing and environment fields', () => {
    vi.stubGlobal('navigator', { userAgent: 'test-browser' })
    const now = vi.spyOn(performance, 'now').mockReturnValue(25)
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm', loadMs: 2, compileMs: 3, inferenceStart: 10,
      inputSummary: 'input', outputSummary: 'output', warnings: [],
    })
    expect(receipt.inferenceMs).toBe(15)
    expect(receipt.environment).toBe('browser: test-browser')
    now.mockRestore()
  })

  it('passes phase receipts through when provided', () => {
    const phases = [
      { name: 'generator', backend: 'wasm' as const, loadMs: 10, compileMs: 20, inferenceMs: 30, warnings: ['w'] },
      { name: 'decoder', backend: 'wasm' as const, loadMs: 5, compileMs: 6 },
    ]
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm', loadMs: 15, compileMs: 26, inferenceStart: 0,
      inputSummary: 'input', outputSummary: 'output', warnings: [],
      phases,
    })
    expect(receipt.phases).toEqual(phases)
  })

  it('omits phases when not provided', () => {
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm', loadMs: 2, compileMs: 3, inferenceStart: 10,
      inputSummary: 'input', outputSummary: 'output', warnings: [],
    })
    expect(receipt.phases).toBeUndefined()
  })

  it('passes diagnostics through when provided', () => {
    const diagnostics = {
      packageName: '@lil-local/test',
      modelId: 'model',
      requestedBackend: 'auto' as const,
      resolvedBackend: 'wasm' as const,
      cacheHit: false,
      fallbackCount: 0,
    }
    const receipt = createInferenceReceipt({
      manifest: { modelId: 'model', version: '1.0.0' },
      backend: 'wasm', loadMs: 2, compileMs: 3, inferenceStart: 10,
      inputSummary: 'input', outputSummary: 'output', warnings: [], diagnostics,
    })
    expect(receipt.diagnostics).toEqual(diagnostics)
  })
})
