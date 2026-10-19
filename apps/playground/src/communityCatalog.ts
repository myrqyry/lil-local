import type { ModelAdapter } from './adapters/types'

const COMMUNITY = 'litert-community'
const PAGE_SIZE = 40
const MODEL_ID = /^litert-community\/([A-Za-z0-9][A-Za-z0-9._-]*)$/

export const COMMUNITY_PAGE_SIZE = PAGE_SIZE

export interface CommunityModel {
  id: string
  name: string
  url: string
  downloads: number | null
  pipelineTag: string | null
  gated: boolean
}

/** Read-only discovery. This URL lists metadata; it never resolves model weights. */
export function communityCatalogUrl(search: string, skip = 0): string {
  if (!Number.isSafeInteger(skip) || skip < 0) {
    throw new Error('Invalid LiteRT Community page offset')
  }

  const url = new URL('https://huggingface.co/api/models')
  url.searchParams.set('author', COMMUNITY)
  url.searchParams.set('sort', 'downloads')
  url.searchParams.set('direction', '-1')
  url.searchParams.set('limit', String(PAGE_SIZE))
  url.searchParams.set('skip', String(skip))
  if (search.trim()) url.searchParams.set('search', search.trim())
  return url.toString()
}

export function parseCommunityModels(payload: unknown): CommunityModel[] {
  if (!Array.isArray(payload)) return []

  const seen = new Set<string>()
  const models: CommunityModel[] = []

  for (const item of payload) {
    if (!item || typeof item !== 'object') continue
    const model = item as Record<string, unknown>
    if (model.private === true || typeof model.id !== 'string') continue

    const match = MODEL_ID.exec(model.id)
    if (!match || match[1] === '.' || match[1] === '..' || seen.has(model.id)) continue
    seen.add(model.id)

    models.push({
      id: model.id,
      name: match[1],
      url: `https://huggingface.co/${COMMUNITY}/${match[1]}`,
      downloads: typeof model.downloads === 'number' && Number.isSafeInteger(model.downloads) && model.downloads >= 0
        ? model.downloads
        : null,
      pipelineTag: typeof model.pipeline_tag === 'string'
        ? model.pipeline_tag.slice(0, 80)
        : null,
      gated: model.gated === true || (typeof model.gated === 'string' && model.gated !== 'false'),
    })
  }
  return models
}

/**
 * A repository match is only evidence that Lil Tart has a model-specific adapter.
 * It is NOT evidence of actual inference; callers must show adapter verification separately.
 */
export function findCommunityAdapter(id: string, adapters: readonly ModelAdapter[]): ModelAdapter | null {
  if (!MODEL_ID.test(id)) return null

  return adapters.find((adapter) => {
    if (!adapter.metadata.modelPath || adapter.isPipeline) return false
    try {
      const url = new URL(adapter.metadata.modelPath)
      if (url.origin !== 'https://huggingface.co') return false
      const match = /^\/(litert-community\/[^/]+)\/resolve\/[^/]+\/.+/.exec(url.pathname)
      return match?.[1] === id
    } catch {
      return false
    }
  }) ?? null
}
