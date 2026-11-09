import type {ModelManifest} from '@lil-local/inference-core'

/**
 * The optimized -web exports are text-only. The separate complete E2B
 * artifact is also compatible with the browser and has real vision/audio
 * inference proof; do not confuse model formats or use an unpinned revision.
 */
export function getPinnedGemma4WebSource(manifest: ModelManifest): string {
  const path = manifest.assets.find(asset => asset.id === 'model')?.path
  const match = path?.match(/^([^/]+\/[^/]+)\/resolve\/([0-9a-f]{40})\/([^/]+-web\.litertlm)$/)
  if (!match || !manifest.modelId.startsWith('gemma-4-') || !manifest.modelId.endsWith('-web')) {
    throw new Error('Gemma 4 Web model requires a pinned revision and a -web.litertlm artifact')
  }
  return `hf://${match[1]}@${match[2]}/${match[3]}`
}

/**
 * The complete E2B artifact is not Android-only: the official
 * LiteRT-LM 0.19.0 browser engine ran real PNG vision and 16k WAV audio
 * against this full model via PodQast -> Lil Local.
 */
export function getPinnedGemma4FullSource(manifest: ModelManifest): string {
  const path = manifest.assets.find(asset => asset.id === 'model')?.path
  const match = path?.match(/^([^/]+\/[^/]+)\/resolve\/([0-9a-f]{40})\/(gemma-4-E2B-it\.litertlm)$/)
  if (!match || manifest.modelId !== 'gemma-4-e2b-it') {
    throw new Error('Full Gemma 4 multimodal model requires a pinned E2B full model artifact')
  }
  return 'hf://' + match[1] + '@' + match[2] + '/' + match[3]
}
