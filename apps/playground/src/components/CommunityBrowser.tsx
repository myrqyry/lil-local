import { useEffect, useState, type FormEvent } from 'react'
import { ArrowRightUpRegular, Loading3Regular, Search2Regular } from '@mingcute/react/core-regular'
import type { ModelAdapter } from '../adapters/types'
import {
  COMMUNITY_PAGE_SIZE,
  communityCatalogUrl,
  findCommunityAdapter,
  parseCommunityModels,
  type CommunityModel,
} from '../communityCatalog'

interface Props {
  adapters: readonly ModelAdapter[]
  onInspect: (adapter: ModelAdapter) => void
}

export default function CommunityBrowser({ adapters, onInspect }: Props) {
  const [searchInput, setSearchInput] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [retry, setRetry] = useState(0)
  const [models, setModels] = useState<CommunityModel[]>([])
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Mounts only when the user opens Community. No automatic model-weight fetches.
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)

    void (async () => {
      try {
        const response = await fetch(communityCatalogUrl(search, page * COMMUNITY_PAGE_SIZE), {
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`Hugging Face returned HTTP ${response.status}`)
        const payload: unknown = await response.json()
        if (!Array.isArray(payload)) throw new Error('Unexpected Hugging Face catalog response')
        if (controller.signal.aborted) return

        const rows = parseCommunityModels(payload)
        setModels((current) => {
          const previous = page === 0 ? [] : current
          const existing = new Set(previous.map((row) => row.id))
          return [...previous, ...rows.filter((row) => !existing.has(row.id))]
        })
        setHasMore(payload.length === COMMUNITY_PAGE_SIZE)
      } catch (err) {
        if (controller.signal.aborted) return
        setError(err instanceof Error ? err.message : 'Could not load the community catalog')
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    })()

    return () => controller.abort()
  }, [search, page, retry])

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setModels([])
    setHasMore(false)
    setPage(0)
    setSearch(searchInput.trim())
    setRetry((current) => current + 1)
  }

  return (
    <section aria-label="LiteRT Community model discovery" className="neo-pod p-3 sm:p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-on-surface">Explore LiteRT Community</h2>
          <p className="mt-0.5 text-xs leading-relaxed text-on-surface-variant">
            Live Hugging Face listings. Discovery is not compatibility or inference verification;
            no model files are downloaded here.
          </p>
        </div>
        <a
          href="https://huggingface.co/litert-community"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-xs text-secondary hover:underline"
        >
          Organization <ArrowRightUpRegular className="h-3.5 w-3.5" />
        </a>
      </div>

      <form onSubmit={submitSearch} className="mt-3 flex gap-2">
        <input
          value={searchInput}
          onChange={(event) => setSearchInput(event.target.value)}
          placeholder="Search community models…"
          aria-label="Search LiteRT Community"
          maxLength={120}
          className="neo-search min-w-0 flex-1 px-3 py-2 text-sm text-on-surface placeholder:text-on-surface-muted"
        />
        <button
          type="submit"
          className="neo-button inline-flex shrink-0 items-center gap-1.5 px-3 py-2 text-xs"
        >
          <Search2Regular className="h-4 w-4" /> Search
        </button>
      </form>

      {error && (
        <div role="alert" className="mt-3 flex flex-wrap items-center gap-2 text-xs text-error">
          <span>{error}</span>
          <button type="button" onClick={() => setRetry((current) => current + 1)} className="underline">
            Retry
          </button>
        </div>
      )}

      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 2xl:grid-cols-3">
        {models.map((model) => {
          const adapter = findCommunityAdapter(model.id, adapters)
          const status = adapter?.disabled
            ? 'Adapter unavailable'
            : adapter
              ? `Adapter · ${adapter.verification?.status?.replace(/-/g, ' ') ?? 'registered'}`
              : 'Discovery only'

          return (
            <article key={model.id} className="rounded-xl border border-outline-variant/65 bg-surface-container-low/70 px-3 py-2.5">
              <div className="flex items-start justify-between gap-2">
                <a
                  href={model.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={model.id}
                  className="min-w-0 break-words text-sm font-semibold leading-tight text-on-surface hover:text-secondary"
                >
                  {model.name} <ArrowRightUpRegular className="inline h-3 w-3" />
                </a>
                {adapter && !adapter.disabled && (
                  <button
                    type="button"
                    className="shrink-0 rounded-md bg-secondary-container px-2 py-1 text-[11px] text-on-secondary-container"
                    onClick={() => onInspect(adapter)}
                  >
                    Inspect
                  </button>
                )}
              </div>
              <p className="mt-1 text-[11px] text-on-surface-muted">
                {model.pipelineTag?.replace(/-/g, ' ') ?? 'Task unspecified'}
                {' · '}{model.downloads === null ? 'Downloads unknown' : `${model.downloads.toLocaleString()} downloads`}
                {model.gated ? ' · gated' : ''}
              </p>
              <p className="mt-1 text-[11px] text-on-surface-variant">{status}</p>
            </article>
          )
        })}
      </div>

      {loading && (
        <p role="status" className="mt-3 flex items-center gap-2 text-xs text-on-surface-muted">
          <Loading3Regular className="h-4 w-4 animate-spin" /> Looking up community models…
        </p>
      )}
      {!loading && !error && models.length === 0 && (
        <p className="mt-3 text-xs text-on-surface-muted">No models found for this search.</p>
      )}
      {!loading && !error && hasMore && (
        <button
          type="button"
          className="neo-button mt-3 px-3 py-2 text-xs"
          onClick={() => setPage((current) => current + 1)}
        >
          Load more models
        </button>
      )}
    </section>
  )
}
