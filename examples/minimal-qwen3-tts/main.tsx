import { createRoot } from 'react-dom/client'
import { useEffect, useState } from 'react'
import { createCachingAssetResolver, createHttpAssetResolver, type AudioInferenceResult, type PipelineProgress } from '@lil-local/inference-core'
import { createLiteRtRuntime } from '@lil-local/runtime-litert'
import { qwen3TtsVariants, Qwen3TtsPipeline } from '@lil-local/qwen3-tts'
import './app.css'

const modelBase = new URL('/models/qwen3-tts/', window.location.href).href

function formatBytes(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`
}

export function App() {
  const [pipeline] = useState(() => new Qwen3TtsPipeline(qwen3TtsVariants.browserMemory, { modelBase }))
  const [status, setStatus] = useState(pipeline.status)
  const [progress, setProgress] = useState<PipelineProgress | null>(null)
  const [backend, setBackend] = useState<string>('detecting')
  const [text, setText] = useState('Hello from LiteRT.')
  const [result, setResult] = useState<AudioInferenceResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const assets = createCachingAssetResolver(createHttpAssetResolver(modelBase))
    const load = async () => {
      try {
        setStatus('loading')
        const context = await createLiteRtRuntime({ assets })
        if (!active) return
        setBackend(context.backend)
        pipeline.onProgress = next => active && setProgress(next)
        await pipeline.load(context)
        if (active) setStatus(pipeline.status)
      } catch (cause) {
        if (!active) return
        setStatus('error')
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    }
    void load()
    return () => {
      active = false
      void pipeline.dispose()
    }
  }, [pipeline])

  const synthesize = async () => {
    setError(null)
    setResult(null)
    const controller = new AbortController()
    try {
      setStatus('running')
      const audio = await pipeline.run({ text }, { maxFrames: 128 }, controller.signal)
      setResult(audio)
      setStatus(pipeline.status)
      const audioContext = new AudioContext()
      const buffer = audioContext.createBuffer(audio.channels, audio.samples.length, audio.sampleRate)
       buffer.copyToChannel(audio.samples as Float32Array<ArrayBuffer>, 0)
      const source = audioContext.createBufferSource()
      source.buffer = buffer
      source.connect(audioContext.destination)
      source.addEventListener('ended', () => void audioContext.close(), { once: true })
      source.start()
    } catch (cause) {
      setStatus(pipeline.status)
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  return (
    <main className="shell">
      <header>
        <p className="eyebrow">LiteRT browser consumer</p>
        <h1>Qwen3-TTS</h1>
        <p className="lede">A standalone runtime proof: real assets, automatic backend selection, and playable audio.</p>
      </header>
      <section className="card facts" aria-label="Model details">
        <div><span>Model</span><strong>{pipeline.manifest.name}</strong></div>
        <div><span>Required download</span><strong>{formatBytes(pipeline.manifest.memory.downloadBytes)}</strong></div>
        <div><span>Backend</span><strong>{backend}</strong></div>
        <div><span>Status</span><strong data-status={status}>{status}</strong></div>
      </section>
      <section className="card">
        <label htmlFor="phrase">Phrase</label>
        <textarea id="phrase" value={text} onChange={event => setText(event.target.value)} rows={3} />
        <button type="button" onClick={() => void synthesize()} disabled={status !== 'ready' || text.trim().length === 0}>
          Synthesize and play
        </button>
        {progress && <p className="muted">{progress.phase}: {progress.step}/{progress.total}</p>}
        {error && <p className="error" role="alert">{error}</p>}
      </section>
      {result && <section className="card receipt">
        <h2>Inference receipt</h2>
        <p>{result.durationSeconds.toFixed(3)} seconds of audio, {result.samples.length} samples</p>
        <dl>
          <dt>Backend</dt><dd>{result.receipt.backend}</dd>
          <dt>Load / compile</dt><dd>{result.receipt.loadMs.toFixed(0)}ms / {result.receipt.compileMs.toFixed(0)}ms</dd>
          <dt>Inference</dt><dd>{result.receipt.inferenceMs.toFixed(0)}ms</dd>
          <dt>Input</dt><dd>{result.receipt.inputSummary}</dd>
          <dt>Output</dt><dd>{result.receipt.outputSummary}</dd>
        </dl>
        {result.receipt.warnings.length > 0 && <p className="muted">Warnings: {result.receipt.warnings.join(', ')}</p>}
      </section>}
    </main>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
