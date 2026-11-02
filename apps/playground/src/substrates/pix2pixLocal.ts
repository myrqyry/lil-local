import type { AssetResolver } from '@lil-local/inference-core'
import { PIX2PIX_EDGE2SHOES_WEIGHT_ASSET } from '@lil-local/pix2pix-webgpu'

function abortError(): Error {
  const error = new Error('Pix2Pix local file read aborted')
  error.name = 'AbortError'
  return error
}

export function createPix2PixFileAssetResolver(file: Blob): AssetResolver {
  return {
    async resolve(asset, options) {
      if (
        asset.id !== PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.id ||
        asset.path !== PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.path
      ) {
        throw new Error(
          `unexpected Pix2Pix asset: ${asset.id} (${asset.path})`,
        )
      }
      if (options?.signal?.aborted) throw abortError()
      const buffer = await file.arrayBuffer()
      if (options?.signal?.aborted) throw abortError()
      options?.onProgress?.({
        asset: asset.id,
        loadedBytes: buffer.byteLength,
        totalBytes: buffer.byteLength,
      })
      return buffer
    },
  }
}
