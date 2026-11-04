import { useEffect, useRef, useState } from 'react'
import { BrowserSubstrateRunner, type BrowserSubstrate } from '@lil-local/inference-core'
import {
  createFaceLandmarkerSubstrate, createGestureRecognizerSubstrate,
  type FaceVisionResult, type GestureVisionResult,
  type VisionFrame, type VisionKind, type VisionProgress,
} from '@lil-local/vision-mediapipe'

type VisionOutput = FaceVisionResult | GestureVisionResult
type Runner = BrowserSubstrateRunner<VisionFrame, VisionOutput, VisionProgress>
const FRAME_INTERVAL_MS = 120

/** This is a consumer of the reusable package, not a second MediaPipe runtime. */
export function MediaPipeVisionPanel() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const runnerRef = useRef<Runner | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRef = useRef<number | null>(null)
  const serialRef = useRef(0)
  const mountedRef = useRef(true)
  const [kind, setKind] = useState<VisionKind>('face')
  const [busy, setBusy] = useState(false)
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('Camera off')
  const [error, setError] = useState('')
  const [result, setResult] = useState<VisionOutput | null>(null)

  const stop = (update = true) => {
    serialRef.current++
    abortRef.current?.abort()
    abortRef.current = null
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    frameRef.current = null
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    const runner = runnerRef.current
    runnerRef.current = null
    if (runner) void runner.dispose().catch(issue => {
      if (update && mountedRef.current) setError(String(issue))
    })
    if (update && mountedRef.current) {
      setBusy(false)
      setRunning(false)
      setStatus('Camera off')
      setResult(null)
    }
  }

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      stop(false)
    }
  }, [])

  const start = async () => {
    if (busy || running) return
    stop()
    const serial = serialRef.current
    const isCurrent = () => mountedRef.current && serial === serialRef.current
    setBusy(true)
    setError('')
    const abort = new AbortController()
    abortRef.current = abort
    const substrate: BrowserSubstrate<VisionFrame, VisionOutput, VisionProgress> = kind === 'face'
      ? createFaceLandmarkerSubstrate({ numFaces: 2 })
      : createGestureRecognizerSubstrate({ numHands: 2 })
    const runner = new BrowserSubstrateRunner(substrate)
    runnerRef.current = runner
    let pendingStream: MediaStream | null = null
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera requires HTTPS or localhost')
      setStatus('Loading official model on demand…')
      await runner.load(progress => {
        if (isCurrent()) setStatus({
          model: 'Downloading the pinned model…',
          integrity: 'Verifying model SHA-256…',
          wasm: 'Starting MediaPipe CPU WASM…',
        }[progress])
      }, abort.signal)
      if (!isCurrent()) return
      setStatus('Requesting camera permission…')
      pendingStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 320 }, height: { ideal: 240 } },
      })
      if (!isCurrent()) return
      const video = videoRef.current
      if (!video) throw new Error('Camera view missing')
      video.srcObject = pendingStream
      await video.play()
      if (!isCurrent()) return
      streamRef.current = pendingStream
      pendingStream = null
      abortRef.current = null
      setBusy(false)
      setRunning(true)
      setStatus('Running locally · camera active')
      let lastFrameAt = -Infinity
      let lastVideoTime = -1
      let framePending = false
      const tick = (now: number) => {
        if (!isCurrent()) return
        frameRef.current = requestAnimationFrame(tick)
        if (document.hidden || framePending || !video.videoWidth ||
            video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
            now - lastFrameAt < FRAME_INTERVAL_MS || video.currentTime === lastVideoTime) return
        lastFrameAt = now
        lastVideoTime = video.currentTime
        framePending = true
        // Only one in-flight inference. BrowserSubstrateRunner owns run/dispose ordering.
        void runner.run({ image: video, timestampMs: now })
          .then(value => { if (isCurrent()) setResult(value) })
          .catch(issue => {
            if (isCurrent()) {
              stop()
              setError(issue instanceof Error ? issue.message : String(issue))
              setStatus('Vision inference failed')
            }
          })
          .finally(() => { framePending = false })
      }
      frameRef.current = requestAnimationFrame(tick)
    } catch (issue) {
      if (isCurrent()) {
        stop()
        if (!(issue instanceof DOMException && issue.name === 'AbortError')) {
          setError(issue instanceof Error ? issue.message : String(issue))
          setStatus('Vision unavailable')
        }
      }
    } finally {
      pendingStream?.getTracks().forEach(track => track.stop())
      if (isCurrent()) setBusy(false)
    }
  }

  const summary = result?.kind === 'face-landmarks'
    ? result.faces.map((face, index) => ({
      key: index,
      title: 'Face ' + (index + 1) + ' · ' + face.landmarks.length + ' landmarks',
      scores: [...face.blendshapes].sort((a, b) => b.score - a.score).slice(0, 6),
    }))
    : result?.kind === 'gestures'
      ? result.hands.map((hand, index) => ({
        key: index,
        title: hand.handedness + ' hand · ' + hand.landmarks.length + ' landmarks',
        scores: hand.gestures.slice(0, 6),
      }))
      : []

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-outline-variant p-3">
        <label htmlFor="vision-task" className="block text-sm font-semibold">Vision task</label>
        <select id="vision-task" value={kind} disabled={busy || running}
          onChange={event => setKind(event.target.value as VisionKind)}
          className="mt-2 w-full rounded-lg border border-outline-variant bg-surface px-3 py-2">
          <option value="face">Face landmarks + blendshapes · 3.8 MB</option>
          <option value="gesture">Hand landmarks + gestures · 8.4 MB</option>
        </select>
        <p className="mt-2 text-sm text-on-surface-variant">
          This playground calls the shared @lil-local/vision-mediapipe package.
          No model downloads or camera access occur until Start.
        </p>
      </div>

      <video ref={videoRef} playsInline muted autoPlay aria-label="Local vision camera"
        className="aspect-[4/3] w-full rounded-xl bg-black object-cover"
        style={{ transform: 'scaleX(-1)' }} />
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={() => void start()} disabled={busy || running}
          className="rounded-xl bg-primary px-4 py-2 font-semibold text-on-primary disabled:opacity-50">
          {busy ? 'Loading…' : 'Start local vision'}
        </button>
        <button type="button" onClick={() => stop()} disabled={!busy && !running}
          className="rounded-xl border border-outline-variant px-4 py-2 disabled:opacity-50">
          Stop and release camera
        </button>
      </div>
      <p role="status" className="text-sm text-on-surface-variant">{status}</p>
      {error && <p role="alert" className="text-sm text-error">{error}</p>}
      <div aria-label="Vision detections" className="space-y-2">
        <p className="text-sm font-semibold">{result ? summary.length + ' detected' : 'No inference results'}</p>
        {summary.map(item => (
          <div key={item.key} className="rounded-lg border border-outline-variant p-3 text-sm">
            <strong>{item.title}</strong>
            <div className="mt-1 flex flex-wrap gap-2">
              {item.scores.map(s => <span key={s.categoryName}>{s.categoryName}: {s.score.toFixed(2)}</span>)}
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-on-surface-variant">
        Frames are processed in this browser, not uploaded by this integration.
        Google MediaPipe may send usage/performance metrics; do not treat this as a
        telemetry-free runtime. Model and WASM require an initial network fetch
        unless the calling app supplies verified local weights and self-hosted WASM.
        Camera permission and tracking are opt-in. CPU inference at ≤8.3 FPS.
      </p>
    </div>
  )
}
