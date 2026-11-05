import { useEffect, useRef, useState } from 'react'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  createUniversalRetrievalSubstrate,
  EMBEDDING_GEMMA2_TEXT_270M,
  type UniversalRetrievalInput,
  type UniversalRetrievalOutput,
} from '@lil-local/retrieval-mediapipe'

type RetrieverRunner = BrowserSubstrateRunner<UniversalRetrievalInput, UniversalRetrievalOutput>

/**
 * The app is a consumer of @lil-local/retrieval-mediapipe.
 * It does not construct another WASM runtime, vector store or embedding engine.
 */
export function MediaPipeRetrievalPanel() {
  const mountedRef = useRef(true)
  const abortRef = useRef<AbortController | null>(null)
  const runnerRef = useRef<RetrieverRunner | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [modelUrl, setModelUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [status, setStatus] = useState('Not loaded')
  const [error, setError] = useState('')
  const [documentId, setDocumentId] = useState('doc-1')
  const [documentText, setDocumentText] = useState('A small cat sitting in a cardboard box.')
  const [image, setImage] = useState<File | null>(null)
  const [query, setQuery] = useState('cat in a box')
  const [result, setResult] = useState<UniversalRetrievalOutput | null>(null)
  const [indexedIds, setIndexedIds] = useState<string[]>([])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      abortRef.current?.abort()
      const runner = runnerRef.current
      runnerRef.current = null
      if (runner) void runner.dispose()
    }
  }, [])

  const load = async () => {
    if (busy || ready || (!file && !modelUrl.trim())) return
    setError(''); setBusy(true)
    const abort = new AbortController()
    abortRef.current = abort
    const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
      model: file ?? modelUrl.trim(),
      // The SDK's original web sample uses the same model for both tasks.
      l2Normalize: true,
      chunkSize: 512,
      chunkOverlap: 100,
    }))
    runnerRef.current = runner
    try {
      setStatus('Loading caller-selected LiteRT-LM model + matching MediaPipe runtime…')
      await runner.load(undefined, abort.signal)
      if (!mountedRef.current || abort.signal.aborted) return
      setReady(true)
      setStatus('Ready · embeddings and search share one loaded model')
    } catch (issue) {
      if (mountedRef.current && !abort.signal.aborted) {
        setError(issue instanceof Error ? issue.message : String(issue))
        setStatus('Retrieval model unavailable')
      }
      if (runnerRef.current === runner) {
        runnerRef.current = null
        await runner.dispose().catch(() => undefined)
      }
    } finally {
      if (abortRef.current === abort) abortRef.current = null
      if (mountedRef.current) setBusy(false)
    }
  }

  const run = async (input: UniversalRetrievalInput) => {
    const runner = runnerRef.current
    if (!runner || !ready || busy) return
    setBusy(true); setError(''); setResult(null)
    try {
      const output = await runner.run(input)
      if (!mountedRef.current) return
      setResult(output)
      const ids = await runner.run({kind:'list-ids'})
      if (ids.kind === 'ids' && mountedRef.current) setIndexedIds(ids.ids)
      setStatus('Local ' + input.kind + ' complete')
    } catch (issue) {
      if (mountedRef.current) {
        setError(issue instanceof Error ? issue.message : String(issue))
        setStatus('Inference failed')
      }
    } finally {
      if (mountedRef.current) setBusy(false)
    }
  }

  const unload = async () => {
    if (busy) return
    const runner = runnerRef.current
    runnerRef.current = null
    setBusy(true)
    try { await runner?.dispose() }
    catch (issue) { if (mountedRef.current) setError(String(issue)) }
    finally {
      if (mountedRef.current) {
        setReady(false)
        setBusy(false)
        setResult(null)
        setIndexedIds([])
        setStatus('Not loaded')
      }
    }
  }

  const runImage = async (kind: 'embed-image' | 'insert-image') => {
    if (!image || busy || !ready) return
    try {
      const bytes = new Uint8Array(await image.arrayBuffer())
      if (!mountedRef.current) return
      if (kind === 'embed-image') await run({kind, imageBytes:bytes})
      else await run({kind, id:documentId.trim(), imageBytes:bytes})
    } catch (issue) {
      if (mountedRef.current) setError(String(issue))
    }
  }

  return <section className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4 text-on-surface">
    <p className="text-sm text-on-surface-variant">
      One model, two tasks: MediaPipe Universal Embedder + Semantic Retriever.
      Text, image and audio embeddings use the same runtime and in-memory index.
      Nothing loads or leaves this browser until you explicitly select a model.
    </p>

    <label className="block text-sm">Local LiteRT-LM retrieval model
      <input aria-label="LiteRT-LM retrieval model file" type="file" accept=".litertlm"
        disabled={busy || ready} onChange={e => setFile(e.target.files?.[0] ?? null)}
        className="mt-1 block w-full text-sm" />
    </label>
    <label className="block text-sm">Or direct model URL (optional)
      <input aria-label="LiteRT-LM retrieval model URL" type="url" value={modelUrl}
        disabled={busy || ready || !!file}
        onChange={e => setModelUrl(e.target.value)}
        placeholder="https://huggingface.co/.../resolve/<revision>/model.litertlm"
        className="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container px-3 py-2 text-sm" />
    </label>
    <button type="button" disabled={busy || ready || !!file}
      onClick={() => setModelUrl(EMBEDDING_GEMMA2_TEXT_270M.url)}
      className="rounded-lg border border-outline-variant px-3 py-2 text-sm disabled:opacity-50">
      Use pinned text-only EmbeddingGemma 2 · 165 MB (select only, no download yet)
    </button>
    <div className="flex flex-wrap gap-2">
      <button type="button" disabled={busy || ready || (!file && !modelUrl.trim())}
        onClick={() => void load()} className="rounded-lg bg-primary px-3 py-2 text-on-primary disabled:opacity-50">
        {busy && !ready ? 'Loading…' : ready ? 'Model ready' : 'Load local retrieval'}
      </button>
      <button type="button" disabled={!ready || busy} onClick={() => void unload()}
        className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
        Unload & clear in-memory index
      </button>
    </div>
    <p role="status" className="text-sm text-on-surface-variant">{status}</p>
    {error && <p role="alert" className="text-sm text-error">{error}</p>}
    <div className="space-y-3 rounded-lg border border-outline-variant p-3">
      <h2 className="font-semibold">Index content once</h2>
      <label className="block text-sm">Record ID
        <input aria-label="Retrieval record ID" value={documentId} onChange={e => setDocumentId(e.target.value)}
          className="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container px-3 py-2" />
      </label>
      <label className="block text-sm">Document text
        <textarea aria-label="Retrieval document text" value={documentText} onChange={e => setDocumentText(e.target.value)}
          rows={3} className="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container px-3 py-2" />
      </label>
      <button type="button" disabled={!ready || busy || !documentText.trim() || !documentId.trim()}
        onClick={() => void run({kind:'insert-text',id:documentId,text:documentText})}
        className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
        Index document
      </button>
      <label className="block text-sm">Optional image (requires a compatible multimodal model)
        <input aria-label="Retrieval image file" type="file" accept="image/*"
          onChange={e => setImage(e.target.files?.[0] ?? null)}
          className="mt-1 block w-full text-sm" />
      </label>
      <div className="flex flex-wrap gap-2">
        <button type="button" disabled={!ready || busy || !image || !documentId.trim()}
          onClick={() => void runImage('insert-image')}
          className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
          Index image
        </button>
        <button type="button" disabled={!ready || busy || !image}
          onClick={() => void runImage('embed-image')}
          className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
          Embed image
        </button>
      </div>
      <p className="text-xs text-on-surface-variant">
        Indexed records in memory: {indexedIds.length} · {indexedIds.join(', ') || 'none'}
      </p>
    </div>

    <div className="space-y-3 rounded-lg border border-outline-variant p-3">
      <h2 className="font-semibold">Query shared embeddings</h2>
      <input aria-label="Retrieval search query" type="text" value={query} onChange={e => setQuery(e.target.value)}
        className="w-full rounded-lg border border-outline-variant bg-surface-container px-3 py-2" />
      <div className="flex flex-wrap gap-2">
        <button type="button" disabled={!ready || busy || !query.trim()}
          onClick={() => void run({kind:'search',query,limit:5})}
          className="rounded-lg bg-primary px-3 py-2 text-on-primary disabled:opacity-50">
          Search locally
        </button>
        <button type="button" disabled={!ready || busy || !query.trim()}
          onClick={() => void run({kind:'embed-text',text:query})}
          className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
          Embed query
        </button>
      </div>
      <div aria-label="MediaPipe retrieval output" className="space-y-2 text-sm">
        {result?.kind === 'search-results' && (result.matches.length
          ? result.matches.map((match,i) => <div key={match.id + i}
              className="rounded-lg border border-outline-variant p-2">
              <strong>{match.id}</strong> · score {match.score.toFixed(3)}
              <p className="text-xs text-on-surface-variant">{match.content.map(part =>
                'text' in part ? part.text : 'imageBytes' in part ? '[image]' : '[audio]').join(' + ')}</p>
            </div>)
          : <p>No matching records.</p>)}
        {result?.kind === 'embedding' && result.embeddings.map((embedding,i) =>
          <div key={i}>Embedding {i + 1} · {result.modality} · {embedding.dimensions} dimensions
            <p className="break-all text-xs text-on-surface-variant">
              {Array.from(embedding.values.slice(0,12)).map(x => x.toFixed(3)).join(', ')}
              {embedding.dimensions > 12 ? ' …' : ''}
            </p>
          </div>)}
      </div>
    </div>
    <p className="text-xs text-on-surface-variant">
      Choose a text-only EmbeddingGemma model for documents, or a compatible
      Text-Vision/Omni model for image/audio. No cloud embedding or implicit
      model fallback. The store is volatile and disappears when you Unload.
      MediaPipe's runtime requires WebGPU and may emit performance telemetry.
    </p>
  </section>
}
