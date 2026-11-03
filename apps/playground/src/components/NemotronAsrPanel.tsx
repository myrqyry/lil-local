import { useEffect, useRef, useState } from 'react'
import { CHUNK_SAMPLES, decodeAudioFile, logMelSpectrogram } from '../substrates/nemotron-web/audio'
import { NemotronClient } from '../substrates/nemotron-web/nemotron-client'
import { chooseNemotronBackend, validateAudioSamples, type NemotronBackend } from '../substrates/nemotronWeb'

const WORKLET_SOURCE = `
class LilLocalPcmCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.buffer = new Float32Array(320); this.length = 0;
    this.port.onmessage = e => {
      if (e.data === 'flush') { this.send(); this.port.postMessage({ flushed: true }); }
    };
  }
  send() {
    if (!this.length) return;
    const samples = this.buffer.slice(0, this.length);
    this.port.postMessage({ samples: samples.buffer }, [samples.buffer]); this.length = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    let offset = 0;
    while (offset < input.length) {
      const n = Math.min(input.length - offset, 320 - this.length);
      this.buffer.set(input.subarray(offset, offset + n), this.length);
      this.length += n; offset += n;
      if (this.length === 320) this.send();
    }
    return true;
  }
}
registerProcessor('lil-local-pcm-capture', LilLocalPcmCapture);
`

type Microphone = {
  stream: MediaStream
  context: AudioContext
  source: MediaStreamAudioSourceNode
  worklet: AudioWorkletNode
}

export function NemotronAsrPanel() {
  const clientRef = useRef<NemotronClient | null>(null)
  const microphoneRef = useRef<Microphone | null>(null)
  const recordingRef = useRef(false)
  const mountedRef = useRef(true)
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [recording, setRecording] = useState(false)
  const [status, setStatus] = useState('Model not loaded. All transcription runs locally in this browser.')
  const [error, setError] = useState('')
  const [transcript, setTranscript] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [backend, setBackend] = useState<NemotronBackend>('auto')
  const [language, setLanguage] = useState(101)

  const setSafeStatus = (message: string) => { if (mountedRef.current) setStatus(message) }
  const setSafeTranscript = (message: string) => { if (mountedRef.current) setTranscript(message) }

  useEffect(() => () => {
    mountedRef.current = false
    recordingRef.current = false
    const microphone = microphoneRef.current
    microphoneRef.current = null
    if (microphone) void closeMicrophone(microphone)
    const client = clientRef.current
    clientRef.current = null
    if (client) void client.dispose().catch(() => undefined)
  }, [])

  const load = async () => {
    if (busy || ready) return
    setBusy(true); setError('')
    const client = new NemotronClient()
    clientRef.current = client
    try {
      const resolved = chooseNemotronBackend('gpu' in navigator, backend)
      setSafeStatus(`Loading ONNX INT4 · requested ${backend}, preferred ${resolved}…`)
      await client.load(setSafeStatus, backend)
      if (!mountedRef.current) return
      setReady(true)
      setSafeStatus('Nemotron loaded. Choose an audio file or record from microphone.')
    } catch (e) {
      setError(errorText(e))
      setSafeStatus('Model not loaded')
      clientRef.current = null
      await client.dispose().catch(() => undefined)
    } finally { if (mountedRef.current) setBusy(false) }
  }

  const transcribeFile = async () => {
    if (!ready || !file || !clientRef.current || busy || recording) return
    setBusy(true); setError(''); setTranscript('')
    try {
      setSafeStatus('Decoding audio to 16 kHz mono…')
      const samples = await decodeAudioFile(file)
      validateAudioSamples(samples)
      const padded = Math.ceil(samples.length / CHUNK_SAMPLES) * CHUNK_SAMPLES
      const mel = logMelSpectrogram(samples, padded)
      setSafeStatus('Transcribing with local ONNX…')
      const text = await clientRef.current.transcribe(mel, language, progress => {
        setSafeTranscript(progress.text)
        setSafeStatus(`Chunk ${progress.chunk}/${progress.totalChunks} · ${Math.round(progress.elapsedMs)} ms`)
      })
      setSafeTranscript(text)
      setSafeStatus('File transcription complete')
    } catch (e) { setError(errorText(e)); setSafeStatus('Transcription failed') }
    finally { if (mountedRef.current) setBusy(false) }
  }

  const startMicrophone = async () => {
    if (!ready || !clientRef.current || busy || recording) return
    setBusy(true); setError(''); setTranscript('')
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone requires HTTPS or localhost')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true } })
      let context: AudioContext | undefined
      try {
        context = new AudioContext({ sampleRate: 16000 })
        if (context.sampleRate !== 16000) throw new Error('AudioContext did not provide 16 kHz capture')
        const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }))
        try { await context.audioWorklet.addModule(url) } finally { URL.revokeObjectURL(url) }
        const source = context.createMediaStreamSource(stream)
        const worklet = new AudioWorkletNode(context, 'lil-local-pcm-capture', { numberOfOutputs: 1, outputChannelCount: [1] })
        microphoneRef.current = { stream, context, source, worklet }
        clientRef.current.startStream(language)
        worklet.port.onmessage = ({ data }: MessageEvent<{ samples?: ArrayBuffer }>) => {
          if (!data.samples || !recordingRef.current) return
          void clientRef.current?.pushAudio(new Float32Array(data.samples), false, setSafeTranscript)
            .then(progress => { if (progress.chunks) setSafeTranscript(progress.text) })
            .catch(e => { setError(errorText(e)); setSafeStatus('Streaming audio failed') })
        }
        source.connect(worklet)
        worklet.connect(context.destination)
        await context.resume()
        recordingRef.current = true
        setRecording(true); setSafeStatus('Listening — live local transcription')
      } catch (e) {
        if (microphoneRef.current) { await closeMicrophone(microphoneRef.current); microphoneRef.current = null }
        else { stream.getTracks().forEach(track => track.stop()); if (context) await context.close().catch(() => undefined) }
        throw e
      }
    } catch (e) { setError(errorText(e)); setSafeStatus('Microphone unavailable') }
    finally { if (mountedRef.current) setBusy(false) }
  }

  const stopMicrophone = async () => {
    const mic = microphoneRef.current
    if (!mic || !recordingRef.current || busy || !clientRef.current) return
    setBusy(true); setRecording(false)
    setSafeStatus('Finishing the last audio chunk…')
    try {
      mic.source.disconnect()
      // Do not discard captured audio on stop: wait for the worklet flush ACK
      // and preserve its final partial PCM message in the worker queue.
      const finalSamples = await new Promise<Float32Array>((resolve, reject) => {
        const tail: Float32Array[] = []
        const timer = setTimeout(() => reject(new Error('Microphone flush timed out')), 3000)
        mic.worklet.port.onmessage = ({ data }: MessageEvent<{ samples?: ArrayBuffer; flushed?: boolean }>) => {
          if (data.samples) tail.push(new Float32Array(data.samples))
          if (data.flushed) {
            clearTimeout(timer)
            const result = new Float32Array(tail.reduce((sum, part) => sum + part.length, 0))
            let offset = 0
            for (const part of tail) { result.set(part, offset); offset += part.length }
            resolve(result)
          }
        }
        mic.worklet.port.postMessage('flush')
      })
      recordingRef.current = false
      const result = await clientRef.current.pushAudio(finalSamples, true, setSafeTranscript)
      setSafeTranscript(result.text)
      setSafeStatus('Microphone transcription complete')
    } catch (e) { setError(errorText(e)); setSafeStatus('Microphone stopped with an error') }
    finally {
      recordingRef.current = false
      microphoneRef.current = null
      await closeMicrophone(mic)
      if (mountedRef.current) setBusy(false)
    }
  }

  return (
    <section className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4 text-on-surface">
      <p className="text-sm text-on-surface-variant">Nemotron 3.5 0.6B · pinned ONNX INT4 · WebGPU / WASM · experimental browser ASR. Model files (~800 MB of browser storage) are fetched only after Load; microphone audio stays on-device.</p>
      <div className="flex flex-wrap gap-3">
        <label className="text-sm">Backend <select aria-label="ASR backend" className="ml-2 rounded border border-outline p-2" disabled={ready || busy} value={backend} onChange={e => setBackend(e.target.value as NemotronBackend)}><option value="auto">Auto: WebGPU → WASM</option><option value="webgpu">WebGPU only</option><option value="wasm">WASM only</option></select></label>
        <label className="text-sm">Language <select aria-label="ASR language" className="ml-2 rounded border border-outline p-2" disabled={busy || recording} value={language} onChange={e => setLanguage(Number(e.target.value))}><option value={101}>Auto detect</option><option value={0}>English</option><option value={4}>Chinese</option><option value={10}>Japanese</option><option value={14}>Korean</option></select></label>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="rounded-lg bg-primary px-4 py-2 text-on-primary disabled:opacity-50" disabled={busy || ready} onClick={() => void load()}>{ready ? 'Model loaded' : busy ? 'Loading…' : 'Load ONNX model'}</button>
        <label className="rounded-lg border border-outline px-4 py-2">Audio file <input aria-label="Audio file" type="file" accept="audio/*,.wav" disabled={busy || recording} onChange={e => setFile(e.target.files?.[0] ?? null)} /></label>
        <button className="rounded-lg border border-outline px-4 py-2 disabled:opacity-50" disabled={!ready || !file || busy || recording} onClick={() => void transcribeFile()}>Transcribe file</button>
        <button className="rounded-lg border border-outline px-4 py-2 disabled:opacity-50" disabled={!ready || (busy && !recording)} onClick={() => void (recording ? stopMicrophone() : startMicrophone())}>{recording ? 'Stop & finalize' : 'Record microphone'}</button>
      </div>
      <p aria-live="polite" className="text-sm text-on-surface-variant">{status}</p>
      {error && <p role="alert" className="text-sm text-error">{error}</p>}
      <div className="min-h-36 rounded-xl bg-surface-container p-3 whitespace-pre-wrap break-words" aria-label="Transcript">{transcript || 'The transcript will appear here as local inference progresses.'}</div>
      {transcript && <button className="rounded border border-outline px-3 py-2 text-sm" onClick={() => void navigator.clipboard.writeText(transcript)}>Copy transcript</button>}
      <p className="text-xs text-on-surface-variant">Source: People-11/Nemotron_Web (MIT); model weights have separate NVIDIA terms. No automatic cloud ASR fallback. Browser inference performance has not yet been qualified here.</p>
    </section>
  )
}

async function closeMicrophone(mic: Microphone): Promise<void> {
  mic.worklet.port.onmessage = null
  mic.stream.getTracks().forEach(track => track.stop())
  mic.worklet.disconnect()
  if (mic.context.state !== 'closed') await mic.context.close().catch(() => undefined)
}
function errorText(error: unknown): string { return error instanceof Error ? error.message : String(error) }
