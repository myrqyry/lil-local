import {
  verifyAssetIntegrity,
  type AssetRequestOptions,
  type AssetResolver,
  type ModelAsset,
  type ModelManifest,
} from '@lil-local/inference-core'

export interface BrowserCacheIdentity {
  modelId: string
  assetId: string
  assetPath: string
  revision: string
}

export interface BrowserCacheStore {
  has?(key: string): Promise<boolean>
  read(key: string): Promise<ArrayBuffer | undefined>
  write(key: string, value: ArrayBuffer): Promise<void>
  delete(key: string): Promise<void>
}

export interface BrowserCacheResolverOptions {
  modelId: string
  revision: string
  manifest: ModelManifest
  inner: AssetResolver
  store?: BrowserCacheStore
}

export interface BrowserCacheAssetResolver extends AssetResolver {
  hasCached(asset: ModelAsset): Promise<boolean>
  invalidate(identity: BrowserCacheIdentity): Promise<void>
  cacheKey(asset: ModelAsset): string
}

function unavailableStore(): BrowserCacheStore {
  return {
    has: async () => { throw new Error('OPFS unavailable') },
    read: async () => { throw new Error('OPFS unavailable') },
    write: async () => { throw new Error('OPFS unavailable') },
    delete: async () => { throw new Error('OPFS unavailable') },
  }
}

export async function createOpfsAssetStore(directoryName = 'litert-playground'): Promise<BrowserCacheStore> {
  const storage = (globalThis as typeof globalThis & {
    navigator?: { storage?: { getDirectory?: () => Promise<FileSystemDirectoryHandle> } }
  }).navigator?.storage
  if (!storage?.getDirectory) return unavailableStore()

  try {
    const root = await storage.getDirectory()
    const directory = await root.getDirectoryHandle(directoryName, { create: true })
    const fileName = (key: string) => encodeURIComponent(key)
    return {
      async has(key) {
        try {
          await directory.getFileHandle(fileName(key))
          return true
        } catch {
          return false
        }
      },
      async read(key) {
        try {
          const file = await directory.getFileHandle(fileName(key))
          return (await file.getFile()).arrayBuffer()
        } catch {
          return undefined
        }
      },
      async write(key, value) {
        const file = await directory.getFileHandle(fileName(key), { create: true })
        const writable = await file.createWritable()
        try {
          await writable.write(value)
        } finally {
          await writable.close()
        }
      },
      async delete(key) {
        try {
          await directory.removeEntry(fileName(key))
        } catch {
          // A missing entry is already deleted.
        }
      },
    }
  } catch {
    return unavailableStore()
  }
}

function assetFromManifest(manifest: ModelManifest, asset: ModelAsset): ModelAsset {
  return manifest.assets.find((candidate) => candidate.id === asset.id) ?? asset
}

export function createBrowserCacheAssetResolver(
  options: BrowserCacheResolverOptions,
): BrowserCacheAssetResolver {
  const storePromise = options.store ? Promise.resolve(options.store) : createOpfsAssetStore()
  const generations = new Map<string, number>()

  function identityFor(asset: ModelAsset): BrowserCacheIdentity {
    return {
      modelId: options.modelId,
      assetId: asset.id,
      assetPath: asset.path,
      revision: options.revision,
    }
  }

  function cacheKey(asset: ModelAsset): string {
    const identity = identityFor(asset)
    return `${identity.modelId}/${identity.assetId}/${identity.assetPath}/${identity.revision}`
  }

  function generation(key: string): number {
    return generations.get(key) ?? 0
  }

  async function hasCached(asset: ModelAsset): Promise<boolean> {
    const manifestAsset = assetFromManifest(options.manifest, asset)
    const key = cacheKey(manifestAsset)
    try {
      const store = await storePromise
      if (store.has) return await store.has(key)
      return (await store.read(key)) !== undefined
    } catch {
      return false
    }
  }

  async function resolve(asset: ModelAsset, requestOptions?: AssetRequestOptions): Promise<ArrayBuffer> {
    const manifestAsset = assetFromManifest(options.manifest, asset)
    const key = cacheKey(manifestAsset)
    const startedGeneration = generation(key)
    let store: BrowserCacheStore
    try {
      store = await storePromise
    } catch {
      store = unavailableStore()
    }

    let cached: ArrayBuffer | undefined
    try {
      cached = await store.read(key)
    } catch {
      store = unavailableStore()
    }
    if (cached !== undefined) return verifyAssetIntegrity(manifestAsset, cached)

    const fresh = await verifyAssetIntegrity(manifestAsset, await options.inner.resolve(asset, requestOptions))
    if (generation(key) === startedGeneration) {
      try {
        await store.write(key, fresh)
      } catch {
        // Cache persistence is optional; inference already has the fresh asset.
      }
    }
    return fresh
  }

  return {
    hasCached,
    resolve,
    stream: async (asset, requestOptions) => {
      const value = await resolve(asset, requestOptions)
      return new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array(value))
          controller.close()
        },
      })
    },
    invalidate: async (identity) => {
      const key = `${identity.modelId}/${identity.assetId}/${identity.assetPath}/${identity.revision}`
      generations.set(key, generation(key) + 1)
      try {
        await (await storePromise).delete(key)
      } catch {
        // Storage may be unavailable; the generation still prevents stale writes.
      }
    },
    cacheKey,
  }
}

export function createAssetObjectUrl(buffer: ArrayBuffer, mimeType?: string): string {
  return URL.createObjectURL(new Blob([buffer], { type: mimeType }))
}

export function revokeAssetObjectUrl(url: string): void {
  URL.revokeObjectURL(url)
}
