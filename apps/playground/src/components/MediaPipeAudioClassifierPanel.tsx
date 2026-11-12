import { useEffect, useRef, useState } from 'react'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  createYamNetSubstrate, type AudioClip, type AudioClassificationResult,
  type AudioLoadProgress,
} from '@lil-local/audio-mediapipe'
import { decodeAudioFile } from '@lil-local/nemotron-asr'

type YamNetRunner = BrowserSubstrateRunner<AudioClip, AudioClassificationResult, AudioLoadProgress>

export function MediaPipeAudioClassifierPanel() {
  const runnerRef = useRef<YamNetRunner | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [status, setStatus] = useState('Not loaded')
  const [error, setError] = useState('')
  const [result, setResult] = useState<AudioClassificationResult | null>(null)

  useEffect(() => () => {
    mountedRef.current = false
    abortRef.current?.abort()
    const runner = runnerRef.current
    runnerRef.current = null
    if (runner) void runner.dispose()
  }, [])

  const load = async () => {
    if (busy || ready) return
    setError(''); setBusy(true)
    const abort = new AbortController()
    abortRef.current = abort
    const runner = new BrowserSubstrateRunner(createYamNetSubstrate({maxResults: 5}))
    runnerRef.current = runner
    try {
      await runner.load(stage => {
        if (mountedRef.current) setStatus({
          model: 'Downloading pinned YAMNet model…',
          integrity: 'Verifying YAMNet SHA-256…',
          wasm: 'Starting MediaPipe Audio WASM…',
        }[stage])
      }, abort.signal)
      if (abort.signal.aborted || !mountedRef.current) return
      setReady(true)
      setStatus('Ready · local audio classifier')
    } catch (issue) {
      if (mountedRef.current && !abort.signal.aborted) {
        setError(issue instanceof Error ? issue.message : String(issue))
        setStatus('Unable to load classifier')
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

  const classify = async () => {
    const runner = runnerRef.current
    if (!runner || !file || busy || !ready) return
    setError(''); setResult(null); setBusy(true)
    try {
      setStatus('Decoding file to 16 kHz mono…')
      const samples = await decodeAudioFile(file)
      if (!mountedRef.current) return
      setStatus('Classifying on-device with YAMNet…')
      const output = await runner.run({samples, sampleRate:16000})
      if (!mountedRef.current) return
      setResult(output)
      setStatus('Classified ' + output.windows.length + ' time window(s)')
    } catch (issue) {
      if (mountedRef.current) {
        setError(issue instanceof Error ? issue.message : String(issue))
        setStatus('Classification failed')
      }
    } finally {
      if (mountedRef.current) setBusy(false)
    }
  }

  const unload = async () => {
    if (busy) return
    abortRef.current?.abort()
    const runner = runnerRef.current
    runnerRef.current = null
    setBusy(true)
    try { await runner?.dispose() }
    catch (issue) { if (mountedRef.current) setError(String(issue)) }
    finally {
      if (mountedRef.current) {
        setReady(false); setBusy(false); setResult(null); setStatus('Not loaded')
      }
    }
  }

  return <section className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4 text-on-surface">
    <p className="text-sm text-on-surface-variant">
      Official YAMNet · local sound classification, 521 audio categories · 4.1 MB verified model.
      Uses shared @lil-local/audio-mediapipe, not a second runtime implementation.
    </p>
    <div className="flex flex-wrap gap-2">
      <button type="button" disabled={busy || ready} onClick={() => void load()}
        className="rounded-lg bg-primary px-3 py-2 text-on-primary disabled:opacity-50">
        {busy && !ready ? 'Loading…' : ready ? 'Loaded' : 'Load verified model'}
      </button>
      <button type="button" disabled={!ready || busy} onClick={() => void unload()}
        className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
        Unload
      </button>
    </div>
    <label className="block text-sm">Audio file
      <input aria-label="Audio file for sound classification" type="file" accept="audio/*,.wav"
        className="mt-1 block w-full text-sm" disabled={busy} onChange={event => setFile(event.target.files?.[0] ?? null)} />
    </label>
    <button type="button" disabled={!ready || busy || !file} onClick={() => void classify()}
      className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">Classify locally</button>
    <p role="status" className="text-sm text-on-surface-variant">{status}</p>
    {error && <p role="alert" className="text-sm text-error">{error}</p>}
    <div aria-label="Sound classification results" className="space-y-2">
      {result?.windows.map((window, i) =>
        <div key={i} className="rounded-lg border border-outline-variant p-2 text-sm">
          <strong>Window {i + 1}{window.timestampMs !== null ? ' · ' + (window.timestampMs / 1000).toFixed(2) + 's' : ''}</strong>
          {window.heads.map((head, index) =>
            <div key={head.headIndex ?? index} className="flex flex-wrap gap-2">
              {head.categories.map(category =>
                <span key={category.index}>{category.categoryName}: {category.score.toFixed(2)}</span>)}
            </div>)}
        </div>)}
      {!result && <p className="text-sm text-on-surface-variant">No results yet.</p>}
    </div>
    <p className="text-xs text-on-surface-variant">
      Loading fetches the official pinned Google YAMNet weights and version-pinned MediaPipe WASM.
      Inference stays in this browser; no microphone is opened. Real audio classes depend on
      the model's training set, and classifications are not guarantees.
    </p>
  </section>
}
