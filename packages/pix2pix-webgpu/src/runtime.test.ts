import { afterEach, describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner, type AssetResolver } from '@lil-local/inference-core'
import {
  createPix2PixEdge2ShoesSubstrateWithDeps,
  type Pix2PixRuntimeDependencies,
} from './runtime'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => { resolve = res })
  return { promise, resolve }
}

function engineFixture(output = new Float32Array(3 * 256 * 256)) {
  return {
    set_normalization_convention: vi.fn(),
    set_raw_pytorch_skip_order: vi.fn(),
    set_use_texture2d_array_skip: vi.fn(),
    run_frame: vi.fn(),
    read_output_nchw: vi.fn(async () => output),
    free: vi.fn(),
  }
}

function setup(engine = engineFixture()) {
  const createEngine = vi.fn(async () => engine)
  const resolveWeights = vi.fn(async () => new ArrayBuffer(16))
  const loadWasm = vi.fn(async () => ({
    Pix2PixEngine: { create_from_uint8_array: createEngine },
    DropoutConvention: { Disabled: 2 },
    NormalizationConvention: { BatchNormEvalFolded: 1 },
  }))
  const deps: Pix2PixRuntimeDependencies = {
    resolveWeights,
    loadWasm,
    now: (() => {
      let value = 100
      return () => value += 5
    })(),
  }
  const assets = {} as AssetResolver
  const canvas = { width: 0, height: 0 } as HTMLCanvasElement
  return { engine, createEngine, resolveWeights, loadWasm, deps, assets, canvas }
}

afterEach(() => vi.unstubAllGlobals())

describe('pix2pix edges2shoes WebGPU substrate', () => {
  it('does nothing until explicit runner load', () => {
    const f = setup()
    const runner = new BrowserSubstrateRunner(
      createPix2PixEdge2ShoesSubstrateWithDeps(
        { assets: f.assets, canvas: f.canvas },
        f.deps,
      ),
    )
    expect(runner.status).toBe('idle')
    expect(f.resolveWeights).not.toHaveBeenCalled()
    expect(f.loadWasm).not.toHaveBeenCalled()
  })

  it('fails closed before asset resolution when WebGPU is unavailable', async () => {
    vi.stubGlobal('navigator', {})
    const f = setup()
    const runner = new BrowserSubstrateRunner(
      createPix2PixEdge2ShoesSubstrateWithDeps(
        { assets: f.assets, canvas: f.canvas },
        f.deps,
      ),
    )
    await expect(runner.load()).rejects.toThrow(/WebGPU/)
    expect(f.resolveWeights).not.toHaveBeenCalled()
    expect(f.loadWasm).not.toHaveBeenCalled()
  })

  it('loads weights, configures the verified engine mode, and returns RGBA output', async () => {
    vi.stubGlobal('navigator', { gpu: {} })
    const f = setup()
    const runner = new BrowserSubstrateRunner(
      createPix2PixEdge2ShoesSubstrateWithDeps(
        { assets: f.assets, canvas: f.canvas },
        f.deps,
      ),
    )
    await runner.load()
    expect(f.resolveWeights).toHaveBeenCalledOnce()
    expect(f.createEngine).toHaveBeenCalledWith(f.canvas, expect.any(Uint8Array), 2)
    expect(f.engine.set_normalization_convention).toHaveBeenCalledWith(1)
    expect(f.engine.set_raw_pytorch_skip_order).toHaveBeenCalledWith(false)
    expect(f.engine.set_use_texture2d_array_skip).toHaveBeenCalledWith(true)

    const result = await runner.run({
      width: 256,
      height: 256,
      data: new Uint8Array(256 * 256 * 4),
    })
    expect(f.engine.run_frame).toHaveBeenCalledOnce()
    expect(f.engine.read_output_nchw).toHaveBeenCalledOnce()
    expect(result.kind).toBe('image')
    expect(result.pixels.slice(0, 4)).toEqual(new Uint8ClampedArray([128, 128, 128, 255]))
    expect(result.receipt.backend).toBe('webgpu')
    expect(result.receipt.modelId).toBe('pix2pix-edges2shoes-unet256')

    await runner.dispose()
    expect(f.engine.free).toHaveBeenCalledOnce()
  })

  it('rejects malformed input before dispatching GPU work', async () => {
    vi.stubGlobal('navigator', { gpu: {} })
    const f = setup()
    const runner = new BrowserSubstrateRunner(
      createPix2PixEdge2ShoesSubstrateWithDeps(
        { assets: f.assets, canvas: f.canvas },
        f.deps,
      ),
    )
    await runner.load()
    await expect(runner.run({
      width: 128,
      height: 256,
      data: new Uint8Array(128 * 256 * 4),
    })).rejects.toThrow(/256x256/)
    expect(f.engine.run_frame).not.toHaveBeenCalled()
    await runner.dispose()
  })

  it('frees an engine that finishes creating after load cancellation', async () => {
    vi.stubGlobal('navigator', { gpu: {} })
    const f = setup()
    const pending = deferred<ReturnType<typeof engineFixture>>()
    f.createEngine.mockImplementationOnce(() => pending.promise)
    const runner = new BrowserSubstrateRunner(
      createPix2PixEdge2ShoesSubstrateWithDeps(
        { assets: f.assets, canvas: f.canvas },
        f.deps,
      ),
    )
    const controller = new AbortController()
    const loading = runner.load(undefined, controller.signal)
    await vi.waitFor(() => expect(f.createEngine).toHaveBeenCalledOnce())
    controller.abort()
    pending.resolve(f.engine)
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(f.engine.free).toHaveBeenCalledOnce()
    expect(runner.status).toBe('idle')
  })
})
