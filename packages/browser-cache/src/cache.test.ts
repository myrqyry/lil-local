import { describe, expect, it, vi } from 'vitest'
import type { AssetProgress, AssetResolver, ModelManifest } from '@lil-local/inference-core'
import {
  createAssetObjectUrl,
  createBrowserCacheAssetResolver,
  revokeAssetObjectUrl,
  type BrowserCacheStore,
} from './cache'

const manifest: ModelManifest = {
  modelId: 'demo-model',
  name: 'Demo model',
  version: '1',
  capabilities: ['image-embedding'],
  backends: { wasm: true },
  memory: { downloadBytes: 3, residentBytes: 3 },
  assets: [{ id: 'model', path: 'model.bin', bytes: 3, mimeType: 'application/octet-stream' }],
}

function asset() {
  return manifest.assets[0]
}

function buffer(...values: number[]): ArrayBuffer {
  return Uint8Array.from(values).buffer
}

function storeWith(values: Map<string, ArrayBuffer>): BrowserCacheStore {
  return {
    read: async (key) => values.get(key),
    write: async (key, value) => { values.set(key, value) },
    delete: async (key) => { values.delete(key) },
  }
}

function innerResolver(resolve: AssetResolver['resolve']): AssetResolver {
  return { resolve }
}

describe('browser cache asset resolver', () => {
  it('returns a stored asset without calling HTTP', async () => {
    const values = new Map<string, ArrayBuffer>()
    const http = vi.fn(async () => buffer(1, 2, 3))
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(http),
      store: storeWith(values),
    })
    values.set(resolver.cacheKey(asset()), buffer(4, 5, 6))

    await expect(resolver.resolve(asset())).resolves.toEqual(buffer(4, 5, 6))
    expect(http).not.toHaveBeenCalled()
  })

  it('falls back to HTTP when storage is unavailable', async () => {
    const http = vi.fn(async () => buffer(1, 2, 3))
    const unavailable: BrowserCacheStore = {
      read: async () => { throw new Error('OPFS unavailable') },
      write: async () => { throw new Error('OPFS unavailable') },
      delete: async () => { throw new Error('OPFS unavailable') },
    }
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(http),
      store: unavailable,
    })

    await expect(resolver.resolve(asset())).resolves.toEqual(buffer(1, 2, 3))
    expect(http).toHaveBeenCalledOnce()
  })

  it('falls back to HTTP on a cache miss', async () => {
    const http = vi.fn(async () => buffer(1, 2, 3))
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(http),
      store: storeWith(new Map()),
    })

    await expect(resolver.resolve(asset())).resolves.toEqual(buffer(1, 2, 3))
    expect(http).toHaveBeenCalledOnce()
  })

  it('uses model, asset, path, and revision in the cache key', () => {
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(async () => buffer(1, 2, 3)),
      store: storeWith(new Map()),
    })

    expect(resolver.cacheKey(asset())).toContain('demo-model')
    expect(resolver.cacheKey(asset())).toContain('model')
    expect(resolver.cacheKey(asset())).toContain('model.bin')
    expect(resolver.cacheKey(asset())).toContain('rev-a')
  })

  it('forwards HTTP progress', async () => {
    const progress: AssetProgress[] = []
    const http = vi.fn(async (_asset, options) => {
      options?.onProgress?.({ asset: 'model', loadedBytes: 3, totalBytes: 3 })
      return buffer(1, 2, 3)
    })
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(http),
      store: storeWith(new Map()),
    })

    await resolver.resolve(asset(), { onProgress: (value) => progress.push(value) })
    expect(progress).toEqual([{ asset: 'model', loadedBytes: 3, totalBytes: 3 }])
  })

  it('rejects a cached buffer with the wrong declared size', async () => {
    const values = new Map<string, ArrayBuffer>()
    const http = vi.fn(async () => buffer(1, 2, 3))
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(http),
      store: storeWith(values),
    })
    values.set(resolver.cacheKey(asset()), buffer(1, 2))

    await expect(resolver.resolve(asset())).rejects.toMatchObject({
      code: 'ASSET_INTEGRITY_FAILED',
    })
    expect(http).not.toHaveBeenCalled()
  })

  it('rejects a cached buffer with the right size but a wrong hash', async () => {
    const values = new Map<string, ArrayBuffer>()
    const http = vi.fn(async () => buffer(1, 2, 3))
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest: {
        ...manifest,
        assets: [{ ...asset(), sha256: 'deadbeef' }],
      },
      inner: innerResolver(http),
      store: storeWith(values),
    })
    values.set(resolver.cacheKey({ ...asset(), sha256: 'deadbeef' }), buffer(4, 5, 6))

    await expect(resolver.resolve(asset())).rejects.toMatchObject({
      code: 'ASSET_INTEGRITY_FAILED',
    })
    expect(http).not.toHaveBeenCalled()
  })

  it('does not repopulate an invalidated entry from stale async work', async () => {
    const values = new Map<string, ArrayBuffer>()
    let release!: (value: ArrayBuffer) => void
    const pending = new Promise<ArrayBuffer>((resolve) => { release = resolve })
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(async () => pending),
      store: storeWith(values),
    })

    const loading = resolver.resolve(asset())
    await resolver.invalidate({
      modelId: 'demo-model',
      assetId: 'model',
      assetPath: 'model.bin',
      revision: 'rev-a',
    })
    release(buffer(1, 2, 3))
    await expect(loading).resolves.toEqual(buffer(1, 2, 3))
    expect(values.has(resolver.cacheKey(asset()))).toBe(false)
  })

  it('invalidates the exact cache identity', async () => {
    const values = new Map<string, ArrayBuffer>()
    const resolver = createBrowserCacheAssetResolver({
      modelId: 'demo-model',
      revision: 'rev-a',
      manifest,
      inner: innerResolver(async () => buffer(1, 2, 3)),
      store: storeWith(values),
    })
    const key = resolver.cacheKey(asset())
    values.set(key, buffer(1, 2, 3))

    await resolver.invalidate({
      modelId: 'demo-model',
      assetId: 'model',
      assetPath: 'model.bin',
      revision: 'rev-a',
    })
    expect(values.has(key)).toBe(false)
  })

  it('creates and revokes object URLs', () => {
    const url = createAssetObjectUrl(buffer(1, 2, 3), 'application/octet-stream')
    expect(url).toMatch(/^blob:/)
    expect(() => revokeAssetObjectUrl(url)).not.toThrow()
  })
})
