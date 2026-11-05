import { useEffect, useRef, useState } from 'react'
import { BrowserSubstrateRunner, type BrowserSubstrate } from '@lil-local/inference-core'
import {
  createFaceLandmarkerSubstrate, createGestureRecognizerSubstrate, createPoseLandmarkerSubstrate,
  createObjectDetectorSubstrate, createImageClassifierSubstrate,
  type FaceVisionResult, type GestureVisionResult, type PoseVisionResult, type ObjectVisionResult, type ImageClassifierVisionResult,
  type VisionFrame, type VisionKind, type VisionProgress,
} from '@lil-local/vision-mediapipe'

type VisionOutput = FaceVisionResult | GestureVisionResult | PoseVisionResult | ObjectVisionResult | ImageClassifierVisionResult
type Runner = BrowserSubstrateRunner<VisionFrame, VisionOutput, VisionProgress>
const FRAME_INTERVAL_MS = 120
// Display-only skeleton connections. The inference package does not own rendering.
const POSE_EDGES: [number, number][] = [
  [11,12], [11,13], [13,15], [12,14], [14,16],
  [11,23], [12,24], [23,24],
  [23,25], [25,27], [24,26], [26,28],
]

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
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageDimensions, setImageDimensions] = useState<{width:number;height:number} | null>(null)
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

  useEffect(() => {
    if (!imageFile) {
      setImagePreview(null)
      return
    }
    const url = URL.createObjectURL(imageFile)
    setImagePreview(url)
    return () => URL.revokeObjectURL(url)
  }, [imageFile])

  const createSelectedSubstrate = (): BrowserSubstrate<VisionFrame, VisionOutput, VisionProgress> =>
    kind === 'face'
      ? createFaceLandmarkerSubstrate({ numFaces: 2 })
      : kind === 'gesture'
        ? createGestureRecognizerSubstrate({ numHands: 2 })
        : kind === 'pose'
          ? createPoseLandmarkerSubstrate({ numPoses: 2 })
          : kind === 'object'
            ? createObjectDetectorSubstrate({ maxResults: 5, scoreThreshold: 0.35 })
            : createImageClassifierSubstrate({ maxResults: 5, scoreThreshold: 0.1 })

  /** Explicit still-image local inference. No webcam permission or uploads. */
  const analyzeImage = async () => {
    if (busy || running || !imageFile) return
    stop()
    const serial = serialRef.current
    const isCurrent = () => mountedRef.current && serial === serialRef.current
    setBusy(true)
    setError('')
    setStatus('Loading model for local image analysis…')
    const abort = new AbortController()
    abortRef.current = abort
    const runner = new BrowserSubstrateRunner(createSelectedSubstrate())
    runnerRef.current = runner
    let bitmap: ImageBitmap | null = null
    try {
      if (!imageFile.type.startsWith('image/')) throw new Error('Choose an image file')
      await runner.load(stage => {
        if (isCurrent()) setStatus({
          model: 'Downloading pinned model…',
          integrity: 'Verifying model SHA-256…',
          wasm: 'Initializing local MediaPipe WASM…',
        }[stage])
      }, abort.signal)
      if (!isCurrent()) return
      setStatus('Decoding and analyzing image locally…')
      bitmap = await createImageBitmap(imageFile)
      if (!isCurrent()) return
      const output = await runner.run({ image: bitmap })
      if (!isCurrent()) return
      setImageDimensions({width: bitmap.width, height: bitmap.height})
      setResult(output)
      setStatus('Image analyzed locally')
    } catch (issue) {
      if (isCurrent() && !(issue instanceof DOMException && issue.name === 'AbortError')) {
        setError(issue instanceof Error ? issue.message : String(issue))
        setStatus('Image analysis failed')
      }
    } finally {
      bitmap?.close()
      if (runnerRef.current === runner) runnerRef.current = null
      if (abortRef.current === abort) abortRef.current = null
      try { await runner.dispose() }
      catch (issue) { if (isCurrent()) setError(String(issue)) }
      if (isCurrent()) setBusy(false)
    }
  }

  const start = async () => {
    if (busy || running) return
    stop()
    const serial = serialRef.current
    const isCurrent = () => mountedRef.current && serial === serialRef.current
    setBusy(true)
    setError('')
    const abort = new AbortController()
    abortRef.current = abort
    const substrate = createSelectedSubstrate()
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
      : result?.kind === 'pose-landmarks'
        ? result.poses.map((pose, index) => ({
          key: index,
          title: 'Pose ' + (index + 1) + ' · ' + pose.landmarks.length +
            ' image points / ' + pose.worldLandmarks.length + ' world points',
          scores: [],
        }))
        : result?.kind === 'object-detections'
          ? result.detections.map((detection, index) => ({
            key: index,
            title: 'Object ' + (index + 1) + (detection.boundingBox
              ? ' · ' + Math.round(detection.boundingBox.width) + ' × ' +
                Math.round(detection.boundingBox.height) + ' px box'
              : ' · no bounding box'),
            scores: detection.categories.slice(0, 3),
          }))
          : result?.kind === 'image-classifications'
            ? result.heads.flatMap(head => ({
              key: head.headIndex,
              title: 'Classification head ' + head.headIndex + (head.headName ? ' · ' + head.headName : ''),
              scores: head.categories,
            }))
            : []

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-outline-variant p-3">
        <label htmlFor="vision-task" className="block text-sm font-semibold">Vision task</label>
        <select id="vision-task" value={kind} disabled={busy || running}
          onChange={event => { setKind(event.target.value as VisionKind); setResult(null) }}
          className="mt-2 w-full rounded-lg border border-outline-variant bg-surface px-3 py-2">
          <option value="face">Face landmarks + blendshapes · 3.8 MB</option>
          <option value="gesture">Hand landmarks + gestures · 8.4 MB</option>
          <option value="pose">Full-body pose + 3D world landmarks · 5.8 MB</option>
          <option value="object">EfficientDet-Lite0 · Object detection · 13.8 MB</option>
          <option value="classifier">EfficientNet-Lite0 · Image classification · 18.6 MB</option>
        </select>
        <p className="mt-2 text-sm text-on-surface-variant">
          This playground calls the shared @lil-local/vision-mediapipe package.
          Model loading is opt-in. Still-image analysis never requests camera access.
        </p>
      </div>

      <label className="block text-sm font-semibold">
        Analyze a still image (no webcam)
        <input aria-label="Image for local vision analysis" type="file" accept="image/*"
          disabled={busy || running}
          onChange={event => {
            setImageFile(event.target.files?.[0] ?? null)
            setImageDimensions(null)
            setResult(null)
          }}
          className="mt-2 block w-full text-sm" />
      </label>
      {imagePreview && <div className="relative inline-block w-full overflow-hidden rounded-xl">
        <img src={imagePreview} alt="Selected image for local inference" className="block w-full" />
        {result?.kind === 'pose-landmarks' && <svg
          viewBox="0 0 1 1" preserveAspectRatio="none"
          className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          {result.poses.map((pose, personIndex) => <g key={personIndex}>
            {POSE_EDGES.map(([a,b]) => {
              const start=pose.landmarks[a]
              const end=pose.landmarks[b]
              if (!start || !end) return null
              return <line key={a+'-'+b} x1={start.x} y1={start.y} x2={end.x} y2={end.y}
                stroke="#e9b16b" strokeWidth={0.003} />
            })}
            {pose.landmarks.map((point,i) => <circle key={i} cx={point.x} cy={point.y} r={0.005}
              fill="#e9b16b" stroke="#28231e" strokeWidth={0.002} />)}
          </g>)}
        </svg>}
        {result?.kind === 'object-detections' && imageDimensions && <svg
          viewBox={`0 0 ${imageDimensions.width} ${imageDimensions.height}`}
          preserveAspectRatio="none"
          className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          {result.detections.map((detected, index) => {
            const box = detected.boundingBox
            if (!box) return null
            const name = detected.categories[0]?.displayName ||
              detected.categories[0]?.categoryName || 'Object'
            const confidence = detected.categories[0]?.score
            const label = name + (confidence === undefined
              ? '' : ' ' + Math.round(confidence * 100) + '%')
            return <g key={index}>
              <rect x={box.originX} y={box.originY} width={box.width} height={box.height}
                fill="none" stroke="#e9b16b" strokeWidth={3} />
              <text x={box.originX + 4} y={Math.max(20, box.originY - 5)}
                fill="#e9b16b" stroke="#28231e" strokeWidth={3}
                paintOrder="stroke" fontSize={18} fontWeight="600">{label}</text>
            </g>
          })}
        </svg>}
      </div>}
      <button type="button" disabled={!imageFile || busy || running}
        onClick={() => void analyzeImage()}
        className="rounded-xl border border-outline-variant px-4 py-2 disabled:opacity-50">
        Analyze selected image locally
      </button>
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
        <p className="text-sm font-semibold">{result ? summary.length + (result.kind === 'image-classifications' ? ' classification heads' : ' detected') : 'No inference results'}</p>
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
