import { useEffect, useRef, useState } from 'react'
import type { AudioTranscriber, TranscriptionSegment } from '@mediapipe/tasks-audio'
import { decodeAudioFile } from '../substrates/nemotron-web/audio'
import {
  PARAKEET_MODEL_FILENAME,
  combineParakeetSegments,
  createParakeetTranscriber,
  createMediaPipeAsrStreamSession,
  type MediaPipeAsrStreamSession,
  parakeetTranscript,
  validateParakeetAudio,
} from '@lil-local/audio-mediapipe'

const CAPTURE_WORKLET = `
class ParakeetCapture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.buffer = new Float32Array(3200)
    this.length = 0
    this.port.onmessage = ({ data }) => {
      if (data === 'flush') {
        this.send()
        this.port.postMessage({ flushed: true })
      }
    }
  }
  send() {
    if (!this.length) return
    const chunk = this.buffer.slice(0, this.length)
    this.port.postMessage({ samples: chunk.buffer }, [chunk.buffer])
    this.length = 0
  }
  process(inputs) {
    const samples = inputs[0]?.[0]
    if (!samples) return true
    for (let offset = 0; offset < samples.length;) {
      const count = Math.min(samples.length - offset, this.buffer.length - this.length)
      this.buffer.set(samples.subarray(offset, offset + count), this.length)
      offset += count
      this.length += count
      if (this.length === this.buffer.length) this.send()
    }
    return true
  }
}
registerProcessor('parakeet-capture', ParakeetCapture)
`

type Microphone = {
  stream: MediaStream
  context: AudioContext
  source: MediaStreamAudioSourceNode
  worklet: AudioWorkletNode
  silent: GainNode
}

async function closeMicrophone(mic: Microphone): Promise<void> {
  mic.worklet.port.onmessage = null
  mic.stream.getTracks().forEach(track => track.stop())
  mic.source.disconnect()
  mic.worklet.disconnect()
  mic.silent.disconnect()
  if (mic.context.state !== 'closed') await mic.context.close().catch(() => undefined)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function ParakeetAsrPanel() {
  const engineRef = useRef<AudioTranscriber | null>(null)
  const microphoneRef = useRef<Microphone | null>(null)
  const mountedRef = useRef(true)
  const acceptingRef = useRef(false)
  const segmentsRef = useRef<TranscriptionSegment[]>([])
  const sessionRef = useRef<MediaPipeAsrStreamSession | null>(null)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [recording, setRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [status, setStatus] = useState('Not loaded')
  const [error, setError] = useState('')

  const report = (message: string) => { if (mountedRef.current) setStatus(message) }
  const reportError = (issue: unknown) => {
    if (mountedRef.current) { setError(errorMessage(issue)); setStatus('Speech recognition failed') }
  }
  const showSegments = (incoming: readonly TranscriptionSegment[]) => {
    segmentsRef.current = combineParakeetSegments(segmentsRef.current, incoming)
    if (mountedRef.current) setTranscript(parakeetTranscript(segmentsRef.current))
  }

  useEffect(() => () => {
    mountedRef.current = false
    acceptingRef.current = false
    const mic = microphoneRef.current
    microphoneRef.current = null
    if (mic) void closeMicrophone(mic)
    const session = sessionRef.current
    sessionRef.current = null
    const engine = engineRef.current
    engineRef.current = null
    if (session) {
      void session.close().then(() => engine?.close(), () => engine?.close())
    } else engine?.close()
  }, [])

  const load = async () => {
    if (busy || ready) return
    setBusy(true); setError('')
    try {
      report(modelFile ? 'Opening local LiteRT-LM model…' : 'Downloading pinned Parakeet weights and MediaPipe runtime…')
      const engine = await createParakeetTranscriber(modelFile ?? undefined)
      if (!mountedRef.current) { engine.close(); return }
      engineRef.current = engine
      try {
        await engine.prewarm()
      } catch (issue) {
        engineRef.current = null
        engine.close()
        throw issue
      }
      setReady(true)
      report('Ready to transcribe locally')
    } catch (issue) { reportError(issue) }
    finally { if (mountedRef.current) setBusy(false) }
  }

  const transcribeFile = async () => {
    if (!engineRef.current || !audioFile || busy || recording) return
    setBusy(true); setError(''); setTranscript(''); segmentsRef.current = []
    try {
      report('Decoding audio into 16 kHz mono…')
      const samples = await decodeAudioFile(audioFile)
      validateParakeetAudio(samples)
      report('Transcribing locally…')
      const result = await engineRef.current.transcribe(samples, true)
      showSegments(result.segments)
      if (!result.segments.length) setTranscript(result.text)
      report('File transcription complete')
    } catch (issue) { reportError(issue) }
    finally { if (mountedRef.current) setBusy(false) }
  }

  const enqueue = (samples: Float32Array) => {
    const session = sessionRef.current
    if (!samples.length || !session) return
    void session.push(samples)
      .then(update => {
        if (mountedRef.current && acceptingRef.current) setTranscript(update.text)
      })
      .catch(issue => {
        if (issue instanceof Error && issue.name === 'AbortError') return
        reportError(issue)
      })
  }

  const startMicrophone = async () => {
    if (!engineRef.current || busy || recording) return
    setBusy(true); setError(''); setTranscript(''); segmentsRef.current = []
    let stream: MediaStream | undefined
    let context: AudioContext | undefined
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone requires HTTPS or localhost')
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true },
      })
      context = new AudioContext({ sampleRate: 16000 })
      if (context.sampleRate !== 16000) throw new Error('AudioContext could not capture at 16 kHz')
      const workletUrl = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: 'text/javascript' }))
      try { await context.audioWorklet.addModule(workletUrl) }
      finally { URL.revokeObjectURL(workletUrl) }
      const source = context.createMediaStreamSource(stream)
      const worklet = new AudioWorkletNode(context, 'parakeet-capture', { numberOfOutputs: 1, outputChannelCount: [1] })
      const silent = context.createGain()
      silent.gain.value = 0
      const mic = { stream, context, source, worklet, silent }
      sessionRef.current = createMediaPipeAsrStreamSession(engineRef.current!, {
        closeEngineOnClose: false,
      })
      microphoneRef.current = mic
      let flushResolve: (() => void) | undefined
      worklet.port.onmessage = ({ data }: MessageEvent<{ samples?: ArrayBuffer; flushed?: boolean }>) => {
        if (data.samples && acceptingRef.current) enqueue(new Float32Array(data.samples))
        if (data.flushed) flushResolve?.()
      }
      // Resolve flush via the same port without discarding the last audio packet.
      ;(mic as Microphone & { finish?: () => Promise<void> }).finish = () => new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => { flushResolve = undefined; reject(new Error('Microphone flush timed out')) }, 3000)
        flushResolve = () => { clearTimeout(timeout); flushResolve = undefined; resolve() }
        worklet.port.postMessage('flush')
      })
      source.connect(worklet)
      worklet.connect(silent)
      silent.connect(context.destination)
      acceptingRef.current = true
      await context.resume()
      setRecording(true)
      report('Listening — speech stays on this device')
      stream = undefined
      context = undefined
    } catch (issue) {
      acceptingRef.current = false
      if (sessionRef.current) { await sessionRef.current.close(); sessionRef.current = null }
      if (microphoneRef.current) { await closeMicrophone(microphoneRef.current); microphoneRef.current = null }
      else { stream?.getTracks().forEach(track => track.stop()); await context?.close().catch(() => undefined) }
      reportError(issue)
    } finally { if (mountedRef.current) setBusy(false) }
  }

  const stopMicrophone = async () => {
    const mic = microphoneRef.current
    const session = sessionRef.current
    if (!mic || !session || busy) return
    setBusy(true); setRecording(false)
    try {
      report('Finalizing captured audio…')
      mic.source.disconnect()
      await (mic as Microphone & { finish?: () => Promise<void> }).finish?.()
      acceptingRef.current = false
      const result = await session.finish()
      if (mountedRef.current) setTranscript(result.text)
      report('Microphone transcription complete')
    } catch (issue) { reportError(issue) }
    finally {
      acceptingRef.current = false
      microphoneRef.current = null
      sessionRef.current = null
      await session.close()
      await closeMicrophone(mic)
      if (mountedRef.current) setBusy(false)
    }
  }

  return (
    <section className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4 text-on-surface">
      <p className="text-sm text-on-surface-variant">MediaPipe Audio Transcriber · Parakeet TDT 0.6B v3 · experimental local ASR. ~615 MB model. CPU inference; no cloud transcription or automatic fallback.</p>
      <label className="block text-sm">Optional local model (.litertlm)
        <input className="mt-1 block w-full text-sm" type="file" accept=".litertlm" disabled={ready || busy} onChange={e => setModelFile(e.target.files?.[0] ?? null)} />
      </label>
      <p className="text-xs text-on-surface-variant">Without a local model, Load fetches {PARAKEET_MODEL_FILENAME} from pinned Hugging Face weights and MediaPipe 1.2.2 WASM from jsDelivr. Downloads are explicit; inference runs locally.</p>
      <div className="flex flex-wrap items-center gap-2">
        <button className="rounded-lg bg-primary px-4 py-2 text-on-primary disabled:opacity-50" disabled={busy || ready} onClick={() => void load()}>{ready ? 'Model loaded' : busy && !ready ? 'Loading…' : 'Load model'}</button>
        <label className="rounded-lg border border-outline px-3 py-2 text-sm">Audio file
          <input className="block max-w-64 text-xs" aria-label="Audio file" type="file" accept="audio/*,.wav" disabled={busy || recording} onChange={e => setAudioFile(e.target.files?.[0] ?? null)} />
        </label>
        <button className="rounded-lg border border-outline px-3 py-2 disabled:opacity-50" disabled={!ready || !audioFile || busy || recording} onClick={() => void transcribeFile()}>Transcribe file</button>
        <button className="rounded-lg border border-outline px-3 py-2 disabled:opacity-50" disabled={!ready || busy} onClick={() => void (recording ? stopMicrophone() : startMicrophone())}>{recording ? 'Stop & finalize' : 'Record microphone'}</button>
      </div>
      <p className="text-sm text-on-surface-variant" aria-live="polite">{status}</p>
      {error && <p className="text-sm text-error" role="alert">{error}</p>}
      <div className="min-h-32 whitespace-pre-wrap break-words rounded-xl bg-surface-container p-3" aria-label="Transcript">{transcript || 'Transcription appears here.'}</div>
      {transcript && <button className="rounded border border-outline px-3 py-2 text-sm" onClick={() => void navigator.clipboard.writeText(transcript)}>Copy transcript</button>}
      <p className="text-xs text-on-surface-variant">Source: Google MediaPipe Tasks (Apache 2.0); NVIDIA Parakeet model (CC BY 4.0). Microphone access is requested only when you start recording.</p>
    </section>
  )
}
