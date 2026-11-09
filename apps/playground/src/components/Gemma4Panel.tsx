import {useEffect, useRef, useState} from 'react'
import {
  createClassicLiteRtLmWorkerClient,
  gemma4E2bManifest,
  gemma4E2bWebManifest,
  gemma4E4bWebManifest,
  getPinnedGemma4WebSource,
  getPinnedGemma4FullSource,
  type LiteRtLmWorkerPrompt,
  type LiteRtLmWorkerClient,
} from '@lil-local/text-gen/litertlm'
import {canonicalWavBlob} from './gemma-audio'

const MODELS = [
  {id: 'full', label: 'Full E2B · Text, image and audio', manifest: gemma4E2bManifest, multimodal: true},
  {id: 'e2b', label: 'Optimized E2B · Text only', manifest: gemma4E2bWebManifest, multimodal: false},
  {id: 'e4b', label: 'Optimized E4B · Text only', manifest: gemma4E4bWebManifest, multimodal: false},
] as const

type Stage = 'idle' | 'loading' | 'ready' | 'generating' | 'error'
type ChatTurn = {role: 'user' | 'assistant'; text: string}
type ChatStats = {tokenCount: number; prefillTokensPerSecond: number; decodeTokensPerSecond: number}

/**
 * Browser-local Gemma 4 native inference through the shared LiteRT-LM
 * worker. The complete E2B model supports real image and WAV audio input
 * with GPU vision + CPU audio (proven in PodQast's Chromium browser run);
 * optimized web exports remain text-only.
 */
export function Gemma4Panel() {
  const [modelId, setModelId] = useState<'full'|'e2b'|'e4b'>('full')
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [attachmentEpoch, setAttachmentEpoch] = useState(0)
  const [prompt, setPrompt] = useState('What is one interesting fact about the night sky?')
  const [maxTokens, setMaxTokens] = useState(128)
  const [stage, setStage] = useState<Stage>('idle')
  const [output, setOutput] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null)
  const [thinking, setThinking] = useState(false)
  const [stats, setStats] = useState<ChatStats | null>(null)
  const [error, setError] = useState('')
  const clientRef = useRef<LiteRtLmWorkerClient | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const generationRef = useRef(0)
  // A native LiteRT-LM Conversation is shared by turns with this ID. Changing
  // it starts a fresh native session on the next prompt; Unload frees the worker.
  const sessionCounterRef = useRef(0)

  const model = MODELS.find(item => item.id === modelId) ?? MODELS[0]
  const modelAsset = model.manifest.assets[0]
  const expectedFileName = modelAsset?.path.split('/').pop() ?? ''
  const remoteBytes = modelAsset?.bytes ?? 0

  function unload() {
    generationRef.current++
    sessionCounterRef.current++
    controllerRef.current?.abort()
    controllerRef.current = null
    clientRef.current?.dispose()
    clientRef.current = null
    setTurns([])
    setActiveQuestion(null)
    setStats(null)
    setImageFile(null)
    setAudioFile(null)
    setAttachmentEpoch(value => value + 1)
    setStage('idle')
  }

  function newChat() {
    if (stage === 'loading' || stage === 'generating') return
    sessionCounterRef.current++
    setTurns([])
    setActiveQuestion(null)
    setStats(null)
    setOutput('')
    setReasoning('')
    setError('')
  }

  useEffect(() => {
    return () => {
      generationRef.current++
      controllerRef.current?.abort()
      clientRef.current?.dispose()
      clientRef.current = null
    }
  }, [])

  async function load() {
    unload()
    const generation = generationRef.current
    setStage('loading')
    setError('')
    setOutput('')
    setReasoning('')
    try {
      const browser = navigator as Navigator & {gpu?: {requestAdapter(): Promise<unknown>}}
      if (!browser.gpu || !(await browser.gpu.requestAdapter())) {
        throw new Error('WebGPU adapter unavailable. Gemma 4 browser models need WebGPU.')
      }
      if (generation !== generationRef.current) return
      if (modelFile && (modelFile.name !== expectedFileName || modelFile.size !== remoteBytes)) {
        throw new Error('Choose the matching official WebGPU model file: ' + expectedFileName)
      }
      const client = createClassicLiteRtLmWorkerClient()
      clientRef.current = client
      await client.load(modelFile ?? (model.multimodal
        ? getPinnedGemma4FullSource(model.manifest)
        : getPinnedGemma4WebSource(model.manifest)), {
        engine: 'litert-lm',
        wasmBase: new URL('/vendor/litert-lm-0.19.0/', window.location.href).href,
        ...(model.multimodal ? {
          visionBackend: 'gpu' as const,
          audioBackend: 'cpu' as const,
          maxNumImages: 8,
        } : {}),
      })
      if (generation !== generationRef.current) return
      setStage('ready')
    } catch (reason) {
      if (generation !== generationRef.current) return
      clientRef.current?.dispose()
      clientRef.current = null
      setStage('error')
      setError(reason instanceof Error ? reason.message : String(reason))
    }
  }

  async function generate() {
    const client = clientRef.current
    if (!client || stage !== 'ready') return
    if (!prompt.trim()) {
      setError('Enter a text prompt.')
      return
    }
    if (!model.multimodal && (imageFile || audioFile)) {
      setError('Choose the full E2B model for image/audio input.')
      return
    }
    if (imageFile && (!['image/png','image/jpeg','image/webp'].includes(imageFile.type) || imageFile.size > 8_000_000)) {
      setError('Image must be PNG, JPEG or WebP (up to 8 MB).')
      return
    }
    const controller = new AbortController()
    controllerRef.current = controller
    const generation = generationRef.current
    const question = prompt.trim()
    const submittedImage = model.multimodal ? imageFile : null
    const submittedAudio = model.multimodal ? audioFile : null
    const sessionId = 'gemma-chat-' + sessionCounterRef.current
    setStage('generating')
    setActiveQuestion(question)
    setError('')
    setOutput('')
    setReasoning('')
    setStats(null)
    try {
      const content: LiteRtLmWorkerPrompt = !submittedImage && !submittedAudio
        ? question
        : [
          {type: 'text', text: question},
          ...(submittedImage ? [{type: 'image' as const, data: submittedImage}] : []),
          ...(submittedAudio ? [{type: 'audio' as const, data: await canonicalWavBlob(submittedAudio)}] : []),
        ]
      const result = await client.generate(
        content,
        text => { if (generation === generationRef.current) setOutput(previous => previous + text) },
        controller.signal,
        {
          maxOutputTokens: maxTokens,
          sessionId,
          ...(thinking ? {enableThinking: true} : {}),
          onReasoning: text => {
            if (generation === generationRef.current && !controller.signal.aborted) {
              setReasoning(previous => previous + text)
            }
          },
          onStats: data => {
            if (generation === generationRef.current && !controller.signal.aborted) setStats(data)
          },
        },
      )
      if (generation === generationRef.current && !controller.signal.aborted) {
        setOutput(result)
        setTurns(previous => [...previous, {role: 'user', text: question}, {role: 'assistant', text: result}])
        setActiveQuestion(null)
        setPrompt('')
        setImageFile(null)
        setAudioFile(null)
        setAttachmentEpoch(value => value + 1)
        setStage('ready')
      }
    } catch (reason) {
      if (generation !== generationRef.current) return
      // A cancelled or failed native turn may have an incomplete KV cache.
      // Never display stale chat history as if the next prompt can use it.
      sessionCounterRef.current++
      setTurns([])
      setActiveQuestion(null)
      setStats(null)
      setOutput('')
      setReasoning('')
      setStage('ready')
      setError(controller.signal.aborted
        ? 'Generation canceled. Native chat context reset.'
        : (reason instanceof Error ? reason.message : String(reason)) + ' · Native chat context reset.')
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }

  return <section className="space-y-4" aria-label="Gemma 4 browser inference">
    <p className="text-sm text-on-surface-variant">
      On-device Gemma 4 through the LiteRT-LM 0.19.0 browser worker.
      Full E2B accepts text, images and canonical WAV audio; optimized E2B/E4B
      variants are text-only. Nothing loads until you choose Load, and Chat stays in memory on this device.
    </p>
    <label className="block text-xs text-on-surface-variant">
      Model
      <select aria-label="Gemma model" value={modelId}
        onChange={event => {
          unload()
          setModelId(event.target.value as 'full'|'e2b'|'e4b')
          setModelFile(null)
          setError('')
        }}
        className="mt-1 w-full rounded-lg border border-outline bg-surface-container px-3 py-2 text-on-surface">
        {MODELS.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
    </label>
    <label className="block text-xs text-on-surface-variant">
      Optional local LiteRT-LM model file (otherwise use the pinned official model)
      <input key={modelId} type="file" accept=".litertlm" aria-label="Gemma local model file"
        onChange={event => {
          unload()
          setModelFile(event.target.files?.[0] ?? null)
          setError('')
        }}
        className="mt-1 block w-full text-xs text-on-surface"/>
    </label>
    <p className="text-xs text-on-surface-variant">
      {modelFile
        ? 'Selected local file: ' + modelFile.name + ' (filename/size checked; content not cryptographically verified by this UI)'
        : 'Official pinned download: ' + (remoteBytes / 1e9).toFixed(2) + ' GB. Download only after clicking Load.'}
    </p>
    <div className="flex flex-wrap items-center gap-3">
      <button type="button" aria-label="Load Gemma locally"
        disabled={stage === 'loading' || stage === 'ready' || stage === 'generating'}
        onClick={() => void load()}
        className="rounded-full bg-primary px-5 py-2 text-sm font-semibold text-on-primary disabled:opacity-50">
        {stage === 'loading' ? 'Loading model…' : modelFile ? 'Load selected local model' : 'Download & load locally'}
      </button>
      {(stage === 'loading' || stage === 'ready' || stage === 'generating') && (
        <button type="button" onClick={unload}
          className="rounded-full border border-outline px-4 py-2 text-xs text-on-surface">
          Unload / stop
        </button>
      )}
      <span className="text-xs text-on-surface-variant" role="status">Gemma status: {stage}</span>
    </div>
    {error && <p role="alert" className="text-sm text-error">{error}</p>}
    <div className="flex flex-wrap items-center gap-4">
      <button type="button" aria-label="New Gemma chat"
        disabled={stage !== 'ready'}
        onClick={newChat}
        className="rounded-full border border-outline px-4 py-2 text-xs text-on-surface disabled:opacity-50">
        New chat
      </button>
      <label className="flex items-center gap-2 text-xs text-on-surface-variant">
        <input type="checkbox" aria-label="Enable thinking" checked={thinking}
          disabled={stage === 'loading' || stage === 'generating'}
          onChange={event => {newChat(); setThinking(event.target.checked)}}
        />
        Enable thinking (starts a new chat)
      </label>
    </div>
    <div aria-label="Gemma chat history" className="space-y-2">
      {(activeQuestion ? turns : turns.slice(0, -1)).map((turn, index) => (
        <div key={index} className="rounded-lg border border-outline-variant p-3 text-sm text-on-surface">
          <span className="font-semibold">{turn.role === 'user' ? 'You' : 'Gemma'}:</span> {turn.text}
        </div>
      ))}
      {activeQuestion && <div className="rounded-lg border border-outline-variant p-3 text-sm text-on-surface">
        <span className="font-semibold">You:</span> {activeQuestion}
      </div>}
    </div>
    {model.multimodal && <div className="grid gap-3 sm:grid-cols-2">
      <label className="block text-xs text-on-surface-variant">
        Image (PNG, JPEG, WebP, up to 8 MB)
        <input key={'image-'+attachmentEpoch} aria-label="Gemma image input"
          disabled={stage === 'generating'} type="file" accept="image/png,image/jpeg,image/webp"
          onChange={event => setImageFile(event.target.files?.[0] ?? null)}
          className="mt-1 block w-full text-xs text-on-surface"/>
        {imageFile && <span>Selected: {imageFile.name}</span>}
      </label>
      <label className="block text-xs text-on-surface-variant">
        Audio (16 kHz mono, 16-bit PCM WAV, up to 8 MB)
        <input key={'audio-'+attachmentEpoch} aria-label="Gemma WAV audio input"
          disabled={stage === 'generating'} type="file" accept=".wav,audio/wav"
          onChange={event => setAudioFile(event.target.files?.[0] ?? null)}
          className="mt-1 block w-full text-xs text-on-surface"/>
        {audioFile && <span>Selected: {audioFile.name}</span>}
      </label>
    </div>}
    <label className="block text-xs text-on-surface-variant">Prompt
      <textarea aria-label="Gemma prompt" rows={3} value={prompt}
        onChange={event => setPrompt(event.target.value)}
        className="mt-1 w-full rounded-xl border border-outline bg-surface-container p-3 text-sm text-on-surface"/>
    </label>
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-xs text-on-surface-variant">
        Max output tokens
        <input aria-label="Gemma max output tokens" type="number" min={1} max={512} value={maxTokens}
          disabled={stage === 'generating' || turns.length > 0}
          onChange={event => setMaxTokens(Math.max(1, Math.min(512, Number(event.target.value)||1)))}
          className="w-20 rounded-lg border border-outline bg-surface-container px-2 py-1 text-on-surface"/>
      </label>
      <button type="button" aria-label="Generate with Gemma locally"
        disabled={stage !== 'ready'} onClick={() => void generate()}
        className="rounded-full bg-primary px-5 py-2 text-sm font-semibold text-on-primary disabled:opacity-50">
        Generate locally
      </button>
      {stage === 'generating' && <button type="button"
        onClick={() => controllerRef.current?.abort()}
        className="rounded-full border border-outline px-4 py-2 text-xs text-on-surface">
        Cancel generation
      </button>}
    </div>
    <div className="rounded-xl border border-outline-variant bg-surface-container p-3">
      <p className="mb-2 text-xs font-semibold text-on-surface-variant">Model output</p>
      <pre aria-label="Gemma model output" className="min-h-24 whitespace-pre-wrap break-words text-sm text-on-surface">{output}</pre>
      {reasoning && <details className="mt-3 text-xs text-on-surface-variant">
        <summary>Reasoning channel</summary>
        <pre className="mt-2 whitespace-pre-wrap break-words">{reasoning}</pre>
      </details>}
    </div>
    {stats && <p aria-label="Gemma inference stats" className="text-xs text-on-surface-variant">
      Native session: {stats.tokenCount.toLocaleString()} tokens · prefill {stats.prefillTokensPerSecond > 0 ? stats.prefillTokensPerSecond.toFixed(1) + ' tok/s' : 'unavailable'} · decode {stats.decodeTokensPerSecond > 0 ? stats.decodeTokensPerSecond.toFixed(1) + ' tok/s' : 'unavailable'}
    </p>}
    <p className="text-xs text-on-surface-variant">
      Full E2B has a verified browser vision path (GPU) and native WAV path
      (CPU audio encoder). Smaller optimized web exports lack those encoders
      and remain text-only. Audio must be a decoded canonical 16 kHz mono PCM
      WAV; this screen does not yet convert MP3 or WebM. Unload releases the
      worker. Model weights are not bundled into the application.
    </p>
  </section>
}
