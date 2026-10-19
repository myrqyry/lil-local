import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RuntimeCapabilities } from '@lil-local/inference-core'
import { AUTO_BACKEND_ORDER, probeRuntimeCapabilities, rankBackends, selectBackend } from './capabilities'

const allAvailable: RuntimeCapabilities = {
  webgpu: { available: true },
  webnn: { available: true },
  wasm: { available: true, simd: false, threads: false, jspi: true },
}

describe('LiteRT backend selection', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('prefers WebGPU, then WebNN, then WASM in auto mode', () => {
    expect(AUTO_BACKEND_ORDER).toEqual(['webgpu', 'webnn', 'wasm'])
    expect(rankBackends(allAvailable)).toEqual(['webgpu', 'webnn', 'wasm'])
    expect(selectBackend(allAvailable)).toBe('webgpu')
  })

  it('respects per-model backend support when ranking fallbacks', () => {
    expect(rankBackends(allAvailable, { webgpu: false })).toEqual(['webnn', 'wasm'])
    expect(rankBackends(allAvailable, { webgpu: false, webnn: false })).toEqual(['wasm'])
  })

  it('does not silently substitute an explicitly requested backend', () => {
    expect(rankBackends(allAvailable, { webgpu: false }, 'webgpu')).toEqual([])
    expect(() => selectBackend(allAvailable, { webgpu: false }, 'webgpu')).toThrow(/No usable backend/)
  })

  it('detects WebNN when navigator.ml is exposed', async () => {
    vi.stubGlobal('navigator', { ml: {} })
    const capabilities = await probeRuntimeCapabilities()
    expect(capabilities.webnn.available).toBe(true)
  })
})

describe('WASM capability probes', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('reports simd:true when validate accepts a SIMD module', async () => {
    vi.stubGlobal('WebAssembly', { validate: () => true })
    const caps = await probeRuntimeCapabilities()
    expect(caps.wasm.simd).toBe(true)
    expect(caps.wasm.available).toBe(true)
  })

  it('reports simd:false when validate rejects the probe module', async () => {
    vi.stubGlobal('WebAssembly', { validate: () => false })
    const caps = await probeRuntimeCapabilities()
    expect(caps.wasm.simd).toBe(false)
  })

  it('reports threads:false when SharedArrayBuffer is absent', async () => {
    vi.stubGlobal('SharedArrayBuffer', undefined as any)
    vi.stubGlobal('WebAssembly', {
      validate: () => true,
      Memory: class {},
    })
    const caps = await probeRuntimeCapabilities()
    expect(caps.wasm.threads).toBe(false)
  })

  it('reports jspi:true only when both promising and Suspending are functions', async () => {
    vi.stubGlobal('WebAssembly', { promising: () => {}, Suspending: class {} })
    const caps = await probeRuntimeCapabilities()
    expect(caps.wasm.jspi).toBe(true)
  })

  it('reports jspi:false when only WebAssembly.Function exists', async () => {
    vi.stubGlobal('WebAssembly', { Function: class {} })
    const caps = await probeRuntimeCapabilities()
    expect(caps.wasm.jspi).toBe(false)
  })
})
