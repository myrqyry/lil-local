import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RuntimePathProof } from './runtimePathProof'
import { downloadRuntimePathReceipt, serializeRuntimePathReceipt } from './runtimePathReceipt'

const proof: RuntimePathProof = {
  modelId: 'tiny-model',
  modelName: 'Tiny Model',
  modelPath: 'models/tiny.tflite',
  selectedBackend: 'auto',
  mainGraphRequestedBackend: 'auto',
  mainGraphResolvedBackend: 'wasm',
  compileDurationMs: 15,
  fallbackCount: 0,
  outputCount: 1,
  graphBackends: [],
  inferenceEvents: [],
  durableVerification: null,
  capturedAt: '2026-10-07T12:00:00.000Z',
}

describe('serializeRuntimePathReceipt', () => {
  it('exports observed session proof while explicitly distinguishing durable verification', () => {
    const receipt = JSON.parse(serializeRuntimePathReceipt(proof))

    expect(receipt).toMatchObject({
      schemaVersion: 1,
      evidenceKind: 'current-session-runtime-path',
      proof: { modelId: 'tiny-model', mainGraphResolvedBackend: 'wasm', durableVerification: null },
    })
    expect(receipt.interpretation).toMatch(/does not promote durable model verification/)
  })
})

describe('downloadRuntimePathReceipt', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('downloads the session receipt as JSON, uses a safe filename, and revokes only after the download has time to start', async () => {
    const click = vi.fn()
    const remove = vi.fn()
    const anchor = { href: '', download: '', click, remove }
    const appendChild = vi.fn()
    const createObjectURL = vi.fn((_blob: Blob) => 'blob:session-receipt')
    const revokeObjectURL = vi.fn()
    let delayedRevoke: (() => void) | undefined
    const setTimeout = vi.fn((callback: () => void, _delay: number) => {
      delayedRevoke = callback
      return 1
    })

    vi.stubGlobal('document', {
      createElement: vi.fn((tag: string) => {
        expect(tag).toBe('a')
        return anchor
      }),
      body: { appendChild },
    })
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
    vi.stubGlobal('window', { setTimeout })

    downloadRuntimePathReceipt({ ...proof, modelId: '../nested:model/id' })

    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
    const saved = createObjectURL.mock.calls[0]![0]
    expect(saved.type).toBe('application/json')
    expect(JSON.parse(await saved.text())).toMatchObject({
      evidenceKind: 'current-session-runtime-path',
      proof: { modelId: '../nested:model/id', durableVerification: null },
    })
    expect(anchor.href).toBe('blob:session-receipt')
    expect(anchor.download).toBe('-nested-model-id-runtime-session.json')
    expect(appendChild).toHaveBeenCalledWith(anchor)
    expect(click).toHaveBeenCalledOnce()
    expect(remove).toHaveBeenCalledOnce()
    expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 60_000)
    expect(revokeObjectURL).not.toHaveBeenCalled()
    delayedRevoke?.()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:session-receipt')
  })

  it('cleans up the download link even when the browser rejects the click', () => {
    const remove = vi.fn()
    const anchor = { href: '', download: '', click: () => { throw new Error('blocked') }, remove }
    const revokeObjectURL = vi.fn()
    let delayedRevoke: (() => void) | undefined
    vi.stubGlobal('document', { createElement: () => anchor, body: { appendChild: vi.fn() } })
    vi.stubGlobal('URL', { createObjectURL: () => 'blob:receipt', revokeObjectURL })
    vi.stubGlobal('window', {
      setTimeout: (callback: () => void) => { delayedRevoke = callback },
    })

    expect(() => downloadRuntimePathReceipt(proof)).toThrow('blocked')
    expect(remove).toHaveBeenCalledOnce()
    delayedRevoke?.()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:receipt')
  })
})
