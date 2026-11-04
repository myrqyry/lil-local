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
  pix2pixEdge2ShoesManifest,
  pix2pixVariant,
  type Pix2PixWeightFormat,
} from '@lil-local/pix2pix-webgpu'

export const PIX2PIX_OPFS_DIRECTORY = 'litert-playground'
// Keep the existing FP32 key stable for users who already installed the original.
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

export function pix2pixCacheKey(format: Pix2PixWeightFormat = 'fp32'): string {
  const { asset, manifest } = pix2pixVariant(format)
  return [
    PIX2PIX_EDGE2SHOES_MODEL_ID,
    asset.id,
    asset.path,
    manifest.version,
  ].join('/')
}

export function pix2pixOpfsLocation(format: Pix2PixWeightFormat = 'fp32'): string {
  return `OPFS /${PIX2PIX_OPFS_DIRECTORY}/${encodeURIComponent(pix2pixCacheKey(format))}`
}

export function createPix2PixFileAssetResolver(
  file: Blob,
  format: Pix2PixWeightFormat = 'fp32',
): AssetResolver {
  const { asset: expected } = pix2pixVariant(format)
  return {
    async resolve(asset, options) {
      if (
        asset.id !== expected.id ||
        asset.path !== expected.path
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
  format: Pix2PixWeightFormat = 'fp32',
): BrowserCacheAssetResolver {
  const { manifest } = pix2pixVariant(format)
  return createBrowserCacheAssetResolver({
    modelId: PIX2PIX_EDGE2SHOES_MODEL_ID,
    revision: manifest.version,
    manifest,
    inner,
    store,
  })
}

function pix2pixCacheIdentity(format: Pix2PixWeightFormat = 'fp32') {
  const { asset, manifest } = pix2pixVariant(format)
  return {
    modelId: PIX2PIX_EDGE2SHOES_MODEL_ID,
    assetId: asset.id,
    assetPath: asset.path,
    revision: manifest.version,
  }
}

export async function hasPix2PixStoredWeights(
  store?: BrowserCacheStore,
  format: Pix2PixWeightFormat = 'fp32',
): Promise<boolean> {
  const { asset } = pix2pixVariant(format)
  return createPix2PixPersistentAssetResolver(undefined, store, format).hasCached(
    asset,
  )
}

export async function importPix2PixWeights(
  file: Blob,
  options?: AssetRequestOptions,
  store?: BrowserCacheStore,
  format: Pix2PixWeightFormat = 'fp32',
): Promise<void> {
  const { asset } = pix2pixVariant(format)
  // Verify the replacement before invalidating a known-good installed copy.
  // A failed import never destroys the existing installed version.
  const selected = await createPix2PixFileAssetResolver(file, format).resolve(
    asset,
    options,
  )
  const verified = await verifyAssetIntegrity(asset, selected)

  const resolver = createPix2PixPersistentAssetResolver(
    { resolve: async () => verified },
    store,
    format,
  )
  await resolver.invalidate(pix2pixCacheIdentity(format))
  await resolver.resolve(asset)

  if (!(await resolver.hasCached(asset))) {
    throw new Error(
      'Pix2Pix weights verified but could not be persisted in Lil Local OPFS storage',
    )
  }
}

export async function removePix2PixStoredWeights(
  store?: BrowserCacheStore,
  format: Pix2PixWeightFormat = 'fp32',
): Promise<void> {
  await createPix2PixPersistentAssetResolver(undefined, store, format).invalidate(
    pix2pixCacheIdentity(format),
  )
}
