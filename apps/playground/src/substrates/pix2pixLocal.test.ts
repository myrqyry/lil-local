import { describe, expect, it, vi } from 'vitest'
import { PIX2PIX_EDGE2SHOES_WEIGHT_ASSET } from '@lil-local/pix2pix-webgpu'
import { createPix2PixFileAssetResolver } from './pix2pixLocal'

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
