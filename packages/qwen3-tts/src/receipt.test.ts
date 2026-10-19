import { describe, expect, it, vi } from 'vitest'
import { Qwen3TtsPipeline } from './pipeline'
import { qwen3TtsVariants } from './manifest'

vi.mock('./phases/generator', () => ({
  GeneratorPhase: class GeneratorPhase {
    name = 'generator'
    loadMs = 1
    compileMs = 2
    inferenceMs = 3
    constructor(variant: unknown, options?: { onProgress?: (p: unknown) => void }) {
      void variant
      void options
    }
    load = vi.fn().mockResolvedValue(undefined)
    generate = vi.fn().mockResolvedValue({
      frames: new Uint16Array([1, 2, 3, 4]),
      frameCount: 1,
      codebooks: 4,
    })
    dispose = vi.fn().mockResolvedValue(undefined)
  },
}))

vi.mock('./phases/decoder', () => ({
  DecoderPhase: class DecoderPhase {
    name = 'decoder'
    loadMs = 4
    compileMs = 5
    inferenceMs = 6
    constructor(variant: unknown, options?: { onProgress?: (p: unknown) => void }) {
      void variant
      void options
    }
    load = vi.fn().mockResolvedValue(undefined)
    decode = vi.fn().mockResolvedValue(new Float32Array([0, 0.1, 0.2]))
    dispose = vi.fn().mockResolvedValue(undefined)
  },
}))

vi.mock('@lil-local/inference-core', async () => ({
  ...(await vi.importActual<typeof import('@lil-local/inference-core')>('@lil-local/inference-core')),
  checkAudioValid: () => [],
}))

describe('Qwen3TtsPipeline receipts', () => {
  it('attaches an automatic receipt with phase receipts to successful audio output', async () => {
    vi.stubGlobal('Worker', undefined)

    const pipeline = new Qwen3TtsPipeline(qwen3TtsVariants.int4)
    await pipeline.load({ backend: 'webgpu', assets: { resolve: vi.fn() }, liteRt: { loadModel: vi.fn(), loadNpy: vi.fn(), fetchBuffer: vi.fn() } })

    const result = await pipeline.run({ text: 'hello' }, { maxFrames: 1 })

    expect(result.kind).toBe('audio')
    expect(result.receipt.modelId).toBe('qwen3-tts-12hz-0.6b-base')
    expect(result.receipt.pipelineVersion).toBe('0.4.0')
    expect(result.receipt.backend).toBe('webgpu')
    expect(result.receipt.inputSummary).toContain('5 characters')
    expect(result.receipt.outputSummary).toContain('24000Hz')
    expect(result.receipt.phases).toHaveLength(2)
    expect(result.receipt.phases?.map((p) => p.name)).toEqual(['generator', 'decoder'])
  })
})
