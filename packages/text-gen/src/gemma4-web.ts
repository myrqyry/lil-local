import type {ModelManifest} from '@lil-local/inference-core'

/**
 * The JavaScript SDK needs a web-compatible model. Do not fall back to
 * the native Android model or an unpinned 'main' revision.
 */
export function getPinnedGemma4WebSource(manifest: ModelManifest): string {
  const path = manifest.assets.find(asset => asset.id === 'model')?.path
  const match = path?.match(/^([^/]+\/[^/]+)\/resolve\/([0-9a-f]{40})\/([^/]+-web\.litertlm)$/)
  if (!match || !manifest.modelId.startsWith('gemma-4-') || !manifest.modelId.endsWith('-web')) {
    throw new Error('Gemma 4 Web model requires a pinned revision and a -web.litertlm artifact')
  }
  return `hf://${match[1]}@${match[2]}/${match[3]}`
}
