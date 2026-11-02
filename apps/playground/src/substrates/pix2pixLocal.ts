import {
  createBrowserCacheAssetResolver,
  type BrowserCacheAssetResolver,
  type BrowserCacheStore,
} from '@lil-local/browser-cache'
import {
  verifyAssetIntegrity,
  type AssetRequestOptions,
  type AssetResolver,
} from '@lil-local/inference-core'
import {
  PIX2PIX_EDGE2SHOES_MODEL_ID,
  PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
  pix2pixEdge2ShoesManifest,
} from '@lil-local/pix2pix-webgpu'

export const PIX2PIX_OPFS_DIRECTORY = 'litert-playground'
export const PIX2PIX_CACHE_REVISION = pix2pixEdge2ShoesManifest.version

function abortError(): Error {
  const error = new Error('Pix2Pix local file read aborted')
  error.name = 'AbortError'
  return error
}

function missingStoredWeightsResolver(): AssetResolver {
  return {
    async resolve() {
      throw new Error(
        'Pix2Pix folded weights are not installed in Lil Local OPFS storage',
      )
    },
  }
}

export function pix2pixCacheKey(): string {
  return [
    PIX2PIX_EDGE2SHOES_MODEL_ID,
    PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.id,
    PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.path,
    PIX2PIX_CACHE_REVISION,
  ].join('/')
}

export function pix2pixOpfsLocation(): string {
  return `OPFS /${PIX2PIX_OPFS_DIRECTORY}/${encodeURIComponent(pix2pixCacheKey())}`
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

export function createPix2PixPersistentAssetResolver(
  inner: AssetResolver = missingStoredWeightsResolver(),
  store?: BrowserCacheStore,
): BrowserCacheAssetResolver {
  return createBrowserCacheAssetResolver({
    modelId: PIX2PIX_EDGE2SHOES_MODEL_ID,
    revision: PIX2PIX_CACHE_REVISION,
    manifest: pix2pixEdge2ShoesManifest,
    inner,
    store,
  })
}

function pix2pixCacheIdentity() {
  return {
    modelId: PIX2PIX_EDGE2SHOES_MODEL_ID,
    assetId: PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.id,
    assetPath: PIX2PIX_EDGE2SHOES_WEIGHT_ASSET.path,
    revision: PIX2PIX_CACHE_REVISION,
  }
}

export async function hasPix2PixStoredWeights(
  store?: BrowserCacheStore,
): Promise<boolean> {
  return createPix2PixPersistentAssetResolver(undefined, store).hasCached(
    PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
  )
}

export async function importPix2PixWeights(
  file: Blob,
  options?: AssetRequestOptions,
  store?: BrowserCacheStore,
): Promise<void> {
  // Verify the replacement before invalidating a known-good installed copy.
  // This keeps a bad picker selection from destroying the working model.
  const selected = await createPix2PixFileAssetResolver(file).resolve(
    PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
    options,
  )
  const verified = await verifyAssetIntegrity(
    PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
    selected,
  )

  const resolver = createPix2PixPersistentAssetResolver(
    { resolve: async () => verified },
    store,
  )
  await resolver.invalidate(pix2pixCacheIdentity())
  await resolver.resolve(PIX2PIX_EDGE2SHOES_WEIGHT_ASSET)

  if (!(await resolver.hasCached(PIX2PIX_EDGE2SHOES_WEIGHT_ASSET))) {
    throw new Error(
      'Pix2Pix weights verified but could not be persisted in Lil Local OPFS storage',
    )
  }
}

export async function removePix2PixStoredWeights(
  store?: BrowserCacheStore,
): Promise<void> {
  await createPix2PixPersistentAssetResolver(undefined, store).invalidate(
    pix2pixCacheIdentity(),
  )
}
