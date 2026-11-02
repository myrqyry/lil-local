import { describe, expect, it, vi } from 'vitest'
import { PIX2PIX_EDGE2SHOES_WEIGHT_ASSET } from '@lil-local/pix2pix-webgpu'
import type { BrowserCacheStore } from '@lil-local/browser-cache'
import {
  PIX2PIX_OPFS_DIRECTORY,
  createPix2PixFileAssetResolver,
  createPix2PixPersistentAssetResolver,
  hasPix2PixStoredWeights,
  importPix2PixWeights,
  pix2pixOpfsLocation,
  removePix2PixStoredWeights,
} from './pix2pixLocal'

describe('Pix2Pix local-file asset resolver', () => {
  it('serves only the pinned folded weight asset and reports complete progress', async () => {
    const bytes = new Uint8Array([1, 2, 3, 4])
    const file = new Blob([bytes])
    const progress = vi.fn()
    const resolver = createPix2PixFileAssetResolver(file)

    const result = await resolver.resolve(PIX2PIX_EDGE2SHOES_WEIGHT_ASSET, {
      onProgress: progress,
    })

    expect([...new Uint8Array(result)]).toEqual([...bytes])
    expect(progress).toHaveBeenLastCalledWith({
      asset: PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.id,
      loadedBytes: 4,
      totalBytes: 4,
    })

    await expect(resolver.resolve({
      ...PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
      id: 'other',
      path: 'other.bin',
    })).rejects.toThrow(/unexpected Pix2Pix asset/)
  })

  it('uses the shared OPFS cache identity and can probe/remove it without reading model bytes', async () => {
    const keys = new Set<string>()
    const reads = vi.fn(async () => undefined)
    const store: BrowserCacheStore = {
      has: async (key) => keys.has(key),
      read: reads,
      write: async (key) => { keys.add(key) },
      delete: async (key) => { keys.delete(key) },
    }
    const resolver = createPix2PixPersistentAssetResolver(undefined, store)
    const key = resolver.cacheKey(PIX2PIX_EDGE2SHOES_WEIGHT_ASSET)
    keys.add(key)

    await expect(hasPix2PixStoredWeights(store)).resolves.toBe(true)
    expect(reads).not.toHaveBeenCalled()
    expect(PIX2PIX_OPFS_DIRECTORY).toBe('litert-playground')
    expect(pix2pixOpfsLocation()).toContain(encodeURIComponent(key))

    await removePix2PixStoredWeights(store)
    await expect(hasPix2PixStoredWeights(store)).resolves.toBe(false)
  })

  it('preserves an installed copy when replacement weights fail integrity', async () => {
    const keys = new Set<string>()
    const store: BrowserCacheStore = {
      has: async (key) => keys.has(key),
      read: async () => undefined,
      write: async (key) => { keys.add(key) },
      delete: async (key) => { keys.delete(key) },
    }
    const key = createPix2PixPersistentAssetResolver(undefined, store)
      .cacheKey(PIX2PIX_EDGE2SHOES_WEIGHT_ASSET)
    keys.add(key)

    await expect(
      importPix2PixWeights(new Blob([new Uint8Array([1, 2, 3])]), undefined, store),
    ).rejects.toMatchObject({ code: 'ASSET_INTEGRITY_FAILED' })

    expect(keys.has(key)).toBe(true)
  })

  it('fails before reading a pre-aborted local file', async () => {
    const file = new Blob([new Uint8Array([1, 2, 3])])
    const arrayBuffer = vi.spyOn(file, 'arrayBuffer')
    const controller = new AbortController()
    controller.abort()

    await expect(
      createPix2PixFileAssetResolver(file).resolve(
        PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
        { signal: controller.signal },
      ),
    ).rejects.toMatchObject({ name: 'AbortError' })
    expect(arrayBuffer).not.toHaveBeenCalled()
  })
})
