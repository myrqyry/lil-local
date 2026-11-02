import { useEffect, useRef, useState } from 'react'
import {
  BrowserSubstrateRunner,
  type ImageInput,
  type InferenceReceipt,
} from '@lil-local/inference-core'
import {
  PIX2PIX_EDGE2SHOES_FOLDED_SHA256,
  PIX2PIX_EDGE2SHOES_WEIGHT_BYTES,
  createPix2PixEdge2ShoesSubstrate,
  type Pix2PixImageResult,
  type Pix2PixLoadProgress,
} from '@lil-local/pix2pix-webgpu'
import {
  createPix2PixPersistentAssetResolver,
  hasPix2PixStoredWeights,
  importPix2PixWeights,
  pix2pixOpfsLocation,
  removePix2PixStoredWeights,
} from '../substrates/pix2pixLocal'
import { buildFrameTimes, buildPreviewFrameTimes, chooseVideoExportMimeType, runFrameSequence } from './pix2pixVideo'

type Runner = BrowserSubstrateRunner<ImageInput, Pix2PixImageResult, Pix2PixLoadProgress>

const VIDEO_PREVIEW_FPS = 8
const VIDEO_PREVIEW_SECONDS = 2
const VIDEO_EXPORT_FPS = 8

interface VideoPreviewStats {
  frames: number
  sourceSeconds: number
  elapsedMs: number
  effectiveFps: number
  averageInferenceMs: number
}

interface VideoExportStats {
  frames: number
  sourceSeconds: number
  elapsedMs: number
  mimeType: string
  bytes: number
}

function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
    : `${(bytes / 1024).toFixed(1)} KiB`
}

function progressLabel(progress: Pix2PixLoadProgress): string {
  if (progress.phase === 'weights') {
    if (progress.loadedBytes !== undefined) {
      const total = progress.totalBytes ?? PIX2PIX_EDGE2SHOES_WEIGHT_BYTES
      return `Reading + verifying weights · ${formatBytes(progress.loadedBytes)} / ${formatBytes(total)}`
    }
    return 'Reading + verifying folded weights…'
  }
  if (progress.phase === 'wasm') return 'Loading bundled Rust/Wasm runtime…'
  return 'Creating WebGPU engine…'
}

async function decodeImageToInput(
  file: File,
  canvas: HTMLCanvasElement,
): Promise<ImageInput> {
  const bitmap = await createImageBitmap(file)
  try {
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) throw new Error('2D canvas is unavailable for image preparation')

    canvas.width = 256
    canvas.height = 256
    const sourceScale = Math.max(256 / bitmap.width, 256 / bitmap.height)
    const sourceWidth = 256 / sourceScale
    const sourceHeight = 256 / sourceScale
    const sourceX = (bitmap.width - sourceWidth) / 2
    const sourceY = (bitmap.height - sourceHeight) / 2

    context.clearRect(0, 0, 256, 256)
    context.drawImage(
      bitmap,
      sourceX,
      sourceY,
      sourceWidth,
      sourceHeight,
      0,
      0,
      256,
      256,
    )
    const image = context.getImageData(0, 0, 256, 256)
    return {
      width: 256,
      height: 256,
      data: new Uint8Array(image.data),
      mimeType: 'image/rgba8',
    }
  } finally {
    bitmap.close()
  }
}

function waitForVideoEvent(
  video: HTMLVideoElement,
  eventName: 'loadedmetadata' | 'seeked',
  signal: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Video preview aborted', 'AbortError'))
      return
    }

    const cleanup = () => {
      video.removeEventListener(eventName, onReady)
      video.removeEventListener('error', onError)
      signal.removeEventListener('abort', onAbort)
    }
    const onReady = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error(video.error ? `Video decode failed (code ${video.error.code})` : 'Video decode failed'))
    }
    const onAbort = () => {
      cleanup()
      reject(new DOMException('Video preview aborted', 'AbortError'))
    }

    video.addEventListener(eventName, onReady, { once: true })
    video.addEventListener('error', onError, { once: true })
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

async function loadVideoSource(
  video: HTMLVideoElement,
  sourceUrl: string,
  signal: AbortSignal,
): Promise<void> {
  const ready = waitForVideoEvent(video, 'loadedmetadata', signal)
  video.src = sourceUrl
  video.load()
  await ready

  if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth || !video.videoHeight) {
    throw new Error('Video metadata is incomplete or unsupported')
  }
}

async function seekVideoFrame(
  video: HTMLVideoElement,
  time: number,
  signal: AbortSignal,
): Promise<void> {
  if (Math.abs(video.currentTime - time) < 0.0005 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return
  const ready = waitForVideoEvent(video, 'seeked', signal)
  video.currentTime = time
  await ready
}

function waitWithAbort(delayMs: number, signal: AbortSignal): Promise<void> {
  if (delayMs <= 0) return Promise.resolve()
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Video export aborted', 'AbortError'))
      return
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, delayMs)
    const onAbort = () => {
      window.clearTimeout(timeout)
      signal.removeEventListener('abort', onAbort)
      reject(new DOMException('Video export aborted', 'AbortError'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function waitForRecorderStop(recorder: MediaRecorder): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      recorder.removeEventListener('stop', onStop)
      recorder.removeEventListener('error', onError)
    }
    const onStop = () => {
      cleanup()
      resolve()
    }
    const onError = (event: Event) => {
      cleanup()
      reject((event as ErrorEvent).error ?? new Error('MediaRecorder failed'))
    }
    recorder.addEventListener('stop', onStop, { once: true })
    recorder.addEventListener('error', onError, { once: true })
  })
}

function captureVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): ImageInput {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('2D canvas is unavailable for video frame preparation')

  canvas.width = 256
  canvas.height = 256
  const sourceScale = Math.max(256 / video.videoWidth, 256 / video.videoHeight)
  const sourceWidth = 256 / sourceScale
  const sourceHeight = 256 / sourceScale
  const sourceX = (video.videoWidth - sourceWidth) / 2
  const sourceY = (video.videoHeight - sourceHeight) / 2

  context.clearRect(0, 0, 256, 256)
  context.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, 256, 256)
  const image = context.getImageData(0, 0, 256, 256)
  return {
    width: 256,
    height: 256,
    data: new Uint8Array(image.data),
    mimeType: 'image/rgba8',
  }
}

export function Pix2PixPanel() {
  const [weightsFile, setWeightsFile] = useState<File | null>(null)
  const [input, setInput] = useState<ImageInput | null>(null)
  const [status, setStatus] = useState('Checking Lil Local model storage…')
  const [error, setError] = useState('')
  const [installed, setInstalled] = useState(false)
  const [checkingStorage, setCheckingStorage] = useState(true)
  const [importing, setImporting] = useState(false)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoRunning, setVideoRunning] = useState(false)
  const [videoExporting, setVideoExporting] = useState(false)
  const [videoStats, setVideoStats] = useState<VideoPreviewStats | null>(null)
  const [videoExportStats, setVideoExportStats] = useState<VideoExportStats | null>(null)
  const [videoExportUrl, setVideoExportUrl] = useState('')
  const [videoExportName, setVideoExportName] = useState('')
  const [stopping, setStopping] = useState(false)
  const [receipt, setReceipt] = useState<InferenceReceipt | null>(null)

  const inputCanvas = useRef<HTMLCanvasElement | null>(null)
  const outputCanvas = useRef<HTMLCanvasElement | null>(null)
  const videoElement = useRef<HTMLVideoElement | null>(null)
  const runner = useRef<Runner | null>(null)
  const pendingRunner = useRef<Runner | null>(null)
  const loadAbort = useRef<AbortController | null>(null)
  const videoAbort = useRef<AbortController | null>(null)
  const videoExportUrlRef = useRef<string | null>(null)
  const generation = useRef(0)

  useEffect(() => {
    let mounted = true
    void hasPix2PixStoredWeights()
      .then(value => {
        if (!mounted) return
        setInstalled(value)
        setStatus(value
          ? 'Model installed in Lil Local OPFS storage. Load it when you are ready.'
          : 'No installed Pix2Pix model yet. Choose the folded .bin once to import it.')
      })
      .catch(reason => {
        if (!mounted) return
        setError('Could not inspect Lil Local model storage: ' + String(reason))
        setStatus('Model storage check failed.')
      })
      .finally(() => {
        if (mounted) setCheckingStorage(false)
      })

    return () => {
      mounted = false
      generation.current += 1
      loadAbort.current?.abort()
      videoAbort.current?.abort()
      const pending = pendingRunner.current
      pendingRunner.current = null
      loadAbort.current = null
      videoAbort.current = null
      const video = videoElement.current
      if (video) {
        video.pause()
        video.removeAttribute('src')
        video.load()
      }
      if (videoExportUrlRef.current) {
        URL.revokeObjectURL(videoExportUrlRef.current)
        videoExportUrlRef.current = null
      }
      if (pending) void pending.dispose().catch(reason => console.error('Pending Pix2Pix cleanup failed:', reason))
      const active = runner.current
      runner.current = null
      if (active) void active.dispose().catch(reason => console.error('Pix2Pix cleanup failed:', reason))
    }
  }, [])

  async function chooseInput(file: File | null) {
    if (!file || !inputCanvas.current) return
    setError('')
    try {
      setInput(await decodeImageToInput(file, inputCanvas.current))
      setReceipt(null)
      setStatus(runner.current ? 'Ready to run.' : 'Input ready. Load the model when you are ready.')
    } catch (reason) {
      setInput(null)
      setError('Could not prepare image: ' + String(reason))
    }
  }

  async function importModel() {
    if (!weightsFile || importing || loading || runner.current) return
    setImporting(true)
    setError('')
    setReceipt(null)
    setStatus(installed
      ? 'Replacing installed Pix2Pix weights in Lil Local OPFS…'
      : 'Importing Pix2Pix weights into Lil Local OPFS…')
    try {
      await importPix2PixWeights(weightsFile, {
        onProgress(progress) {
          setStatus(
            `Reading local weights · ${formatBytes(progress.loadedBytes)} / ${formatBytes(progress.totalBytes ?? weightsFile.size)}`,
          )
        },
      })
      setInstalled(true)
      setStatus('Model installed in Lil Local OPFS storage. You can load it without choosing the file again.')
    } catch (reason) {
      setInstalled(await hasPix2PixStoredWeights())
      setStatus('Model import failed.')
      setError(String(reason))
    } finally {
      setImporting(false)
    }
  }

  async function removeInstalledModel() {
    if (!installed || loading || running || videoRunning || videoExporting || importing || runner.current) return
    setError('')
    setReceipt(null)
    setStatus('Removing installed Pix2Pix weights from Lil Local OPFS…')
    try {
      await removePix2PixStoredWeights()
      setInstalled(false)
      setWeightsFile(null)
      setStatus('Installed Pix2Pix model removed. Choose the folded .bin to import it again.')
    } catch (reason) {
      setError('Could not remove installed Pix2Pix model: ' + String(reason))
      setStatus('Model removal failed.')
    }
  }

  async function loadModel() {
    if (!installed || !outputCanvas.current || loading || importing || runner.current) {
      if (!installed) setError('Import the folded edges2shoes model into Lil Local storage first.')
      return
    }

    const requestId = ++generation.current
    const controller = new AbortController()
    const localRunner = new BrowserSubstrateRunner(
      createPix2PixEdge2ShoesSubstrate({
        assets: createPix2PixPersistentAssetResolver(),
        canvas: outputCanvas.current,
      }),
    )
    loadAbort.current = controller
    pendingRunner.current = localRunner
    setLoading(true)
    setError('')
    setReceipt(null)
    setStatus('Starting explicit local WebGPU load…')

    try {
      await localRunner.load(progress => {
        if (requestId === generation.current) setStatus(progressLabel(progress))
      }, controller.signal)
      if (requestId !== generation.current || controller.signal.aborted) {
        await localRunner.dispose()
        return
      }
      if (pendingRunner.current === localRunner) pendingRunner.current = null
      runner.current = localRunner
      setStatus('Ready · pinned folded weights verified · WebGPU engine loaded.')
    } catch (reason) {
      if (requestId === generation.current) {
        if (controller.signal.aborted) {
          setStatus('Load stopped; model not loaded.')
          if ((reason as Error)?.name !== 'AbortError') {
            setError('Loading also failed: ' + String(reason))
          }
        } else {
          setStatus('Load failed.')
          setError(String(reason))
        }
      }
    } finally {
      if (pendingRunner.current === localRunner) pendingRunner.current = null
      if (loadAbort.current === controller) loadAbort.current = null
      if (requestId === generation.current) {
        setLoading(false)
        setStopping(false)
      }
    }
  }

  async function stopLoading() {
    const pending = pendingRunner.current
    if (!pending || stopping) return
    setStopping(true)
    setStatus('Stopping; waiting for current WebGPU setup step and cleanup…')
    loadAbort.current?.abort()
    try {
      await pending.dispose()
    } catch (reason) {
      setError('Runtime cleanup failed: ' + String(reason))
      setStatus('Load cleanup failed.')
    }
  }

  async function unload() {
    if (loading || running || videoRunning || videoExporting) return
    ++generation.current
    const active = runner.current
    runner.current = null
    try {
      if (active) await active.dispose()
      setReceipt(null)
      setError('')
      setStatus('Unloaded. Installed OPFS model remains available.')
    } catch (reason) {
      setError('Runtime disposal failed: ' + String(reason))
    }
  }

  async function run() {
    const active = runner.current
    if (!active || !input || loading || running || videoRunning || videoExporting) return
    setRunning(true)
    setError('')
    setReceipt(null)
    setStatus('Running full Pix2Pix U-Net on WebGPU…')
    try {
      const result = await active.run(input)
      setReceipt(result.receipt)
      setStatus('Inference complete. Output is rendered below.')
    } catch (reason) {
      setStatus('Inference failed.')
      setError(String(reason))
    } finally {
      setRunning(false)
    }
  }

  function clearVideoExport() {
    if (videoExportUrlRef.current) {
      URL.revokeObjectURL(videoExportUrlRef.current)
      videoExportUrlRef.current = null
    }
    setVideoExportUrl('')
    setVideoExportName('')
    setVideoExportStats(null)
  }

  async function runVideoPreview() {
    const active = runner.current
    const video = videoElement.current
    const canvas = inputCanvas.current
    if (!active || !video || !canvas || !videoFile || loading || running || videoRunning || videoExporting) return

    const controller = new AbortController()
    const requestId = generation.current
    const sourceUrl = URL.createObjectURL(videoFile)
    let completed = 0
    let inferenceTotalMs = 0
    let lastReceipt: InferenceReceipt | null = null

    videoAbort.current = controller
    setVideoRunning(true)
    setVideoStats(null)
    setError('')
    setReceipt(null)
    setStatus('Reading local video metadata…')

    try {
      await loadVideoSource(video, sourceUrl, controller.signal)
      const times = buildPreviewFrameTimes(video.duration, VIDEO_PREVIEW_FPS, VIDEO_PREVIEW_SECONDS)
      if (!times.length) throw new Error('Video contains no previewable frames')

      const sourceSeconds = Math.min(video.duration, VIDEO_PREVIEW_SECONDS)
      const startedAt = performance.now()

      await runFrameSequence({
        times,
        signal: controller.signal,
        async processFrame(time) {
          await seekVideoFrame(video, time, controller.signal)
          const frame = captureVideoFrame(video, canvas)
          const result = await active.run(frame)
          lastReceipt = result.receipt
          inferenceTotalMs += result.receipt.inferenceMs
        },
        onProgress(progress) {
          completed = progress.completed
          if (requestId === generation.current) {
            setStatus(
              `Video preview · frame ${progress.completed}/${progress.total} · source ${progress.time.toFixed(2)} s`,
            )
          }
        },
      })

      const elapsedMs = performance.now() - startedAt
      const effectiveFps = completed / (elapsedMs / 1000)
      if (requestId === generation.current) {
        if (lastReceipt) setReceipt(lastReceipt)
        setVideoStats({
          frames: completed,
          sourceSeconds,
          elapsedMs,
          effectiveFps,
          averageInferenceMs: inferenceTotalMs / completed,
        })
        setStatus(
          `Video preview complete · ${completed} frames · ${effectiveFps.toFixed(1)} effective fps.`,
        )
      }
    } catch (reason) {
      if (requestId === generation.current) {
        if ((reason as Error)?.name === 'AbortError') {
          setStatus(`Video preview stopped after ${completed} frame${completed === 1 ? '' : 's'}.`)
        } else {
          setStatus('Video preview failed.')
          setError(String(reason))
        }
      }
    } finally {
      video.pause()
      video.removeAttribute('src')
      video.load()
      URL.revokeObjectURL(sourceUrl)
      if (videoAbort.current === controller) videoAbort.current = null
      if (requestId === generation.current) setVideoRunning(false)
    }
  }

  async function runVideoExport() {
    const active = runner.current
    const video = videoElement.current
    const inputCanvasNode = inputCanvas.current
    const outputCanvasNode = outputCanvas.current
    if (!active || !video || !inputCanvasNode || !outputCanvasNode || !videoFile ||
        loading || running || videoRunning || videoExporting) return

    if (typeof MediaRecorder === 'undefined') {
      setError('MediaRecorder is unavailable in this browser.')
      return
    }

    const controller = new AbortController()
    const requestId = generation.current
    const sourceUrl = URL.createObjectURL(videoFile)
    let completed = 0
    let stream: MediaStream | null = null
    let recorder: MediaRecorder | null = null
    let recorderStopped: Promise<void> | null = null
    const chunks: Blob[] = []

    videoAbort.current = controller
    setVideoExporting(true)
    clearVideoExport()
    setError('')
    setReceipt(null)
    setStatus('Reading local video metadata for WebM export…')

    try {
      await loadVideoSource(video, sourceUrl, controller.signal)
      const times = buildFrameTimes(video.duration, VIDEO_EXPORT_FPS)
      if (!times.length) throw new Error('Video contains no exportable frames')

      const mimeType = chooseVideoExportMimeType(type => MediaRecorder.isTypeSupported(type))
      stream = outputCanvasNode.captureStream(0)
      const track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined
      if (!track || typeof track.requestFrame !== 'function') {
        throw new Error('Manual canvas frame capture is unavailable in this browser')
      }

      recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 2_500_000,
      })
      recorder.addEventListener('dataavailable', event => {
        if (event.data.size > 0) chunks.push(event.data)
      })
      recorderStopped = waitForRecorderStop(recorder)
      recorder.start()

      const startedAt = performance.now()
      await runFrameSequence({
        times,
        signal: controller.signal,
        async processFrame(time, index) {
          await seekVideoFrame(video, time, controller.signal)
          const frame = captureVideoFrame(video, inputCanvasNode)
          await active.run(frame)
          track.requestFrame()

          const targetElapsedMs = ((index + 1) / VIDEO_EXPORT_FPS) * 1000
          const remainingMs = targetElapsedMs - (performance.now() - startedAt)
          await waitWithAbort(remainingMs, controller.signal)
        },
        onProgress(progress) {
          completed = progress.completed
          if (requestId === generation.current) {
            setStatus(
              `Exporting WebM · frame ${progress.completed}/${progress.total} · source ${progress.time.toFixed(2)} s`,
            )
          }
        },
      })

      // MediaRecorder timestamps frames at capture time. Stamp one terminal hold
      // request at the source end; exact container duration remains browser-controlled.
      track.requestFrame()

      if (recorder.state !== 'inactive') recorder.stop()
      await recorderStopped

      const blob = new Blob(chunks, { type: mimeType })
      if (!blob.size) throw new Error('WebM recorder produced an empty file')

      const exportUrl = URL.createObjectURL(blob)
      videoExportUrlRef.current = exportUrl
      const baseName = videoFile.name.replace(/\.[^.]+$/, '') || 'pix2pix-video'
      const exportName = `${baseName}-pix2pix-${VIDEO_EXPORT_FPS}fps.webm`
      const elapsedMs = performance.now() - startedAt

      if (requestId === generation.current) {
        setVideoExportUrl(exportUrl)
        setVideoExportName(exportName)
        setVideoExportStats({
          frames: completed,
          sourceSeconds: video.duration,
          elapsedMs,
          mimeType,
          bytes: blob.size,
        })
        setStatus(
          `Video export ready · ${completed} frames · ${(blob.size / (1024 * 1024)).toFixed(1)} MiB WebM.`,
        )
      }
    } catch (reason) {
      if (requestId === generation.current) {
        if ((reason as Error)?.name === 'AbortError') {
          setStatus(`Video export stopped after ${completed} frame${completed === 1 ? '' : 's'}; partial file discarded.`)
        } else {
          setStatus('Video export failed.')
          setError(String(reason))
        }
      }
    } finally {
      if (recorder && recorder.state !== 'inactive') {
        try {
          recorder.stop()
          await recorderStopped
        } catch {
          // Preserve the original export error; cleanup is best-effort here.
        }
      }
      for (const track of stream?.getTracks() ?? []) track.stop()
      video.pause()
      video.removeAttribute('src')
      video.load()
      URL.revokeObjectURL(sourceUrl)
      if (videoAbort.current === controller) videoAbort.current = null
      if (requestId === generation.current) setVideoExporting(false)
    }
  }

  function stopVideoPreview() {
    if (!videoRunning) return
    setStatus('Stopping video preview after the current WebGPU frame…')
    videoAbort.current?.abort()
  }

  function stopVideoExport() {
    if (!videoExporting) return
    setStatus('Stopping video export after the current WebGPU frame…')
    videoAbort.current?.abort()
  }

  const loaded = !!runner.current

  return (
    <section className="space-y-4 rounded-2xl border border-outline bg-surface-container p-4 text-on-surface">
      <div className="space-y-1">
        <p className="text-sm font-semibold">Official edges2shoes · U-Net 256 · custom Rust/Wasm WebGPU</p>
        <p className="text-sm text-on-surface-variant">
          Import the verified model once into Lil Local's persistent OPFS store, then future sessions load that installed copy. Nothing downloads automatically and there is no CPU, WebGL, or cloud fallback.
        </p>
        <p className="text-xs text-on-surface-variant">
          Expected folded weights: {formatBytes(PIX2PIX_EDGE2SHOES_WEIGHT_BYTES)} · SHA-256 {PIX2PIX_EDGE2SHOES_FOLDED_SHA256.slice(0, 16)}…
        </p>
        <div className="rounded-xl border border-outline bg-surface/50 p-3 text-xs text-on-surface-variant">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-on-surface">
              {checkingStorage ? 'Checking model storage…' : installed ? 'Installed in browser storage' : 'Not installed'}
            </span>
            <span>{installed ? formatBytes(PIX2PIX_EDGE2SHOES_WEIGHT_BYTES) : '0 B stored'}</span>
          </div>
          <code className="mt-2 block break-all text-[10px] text-on-surface-muted">{pix2pixOpfsLocation()}</code>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-2 text-sm">
          <span className="font-medium">1 · Import / replace folded model weights</span>
          <input
            type="file"
            accept=".bin,application/octet-stream"
            disabled={checkingStorage || importing || loading || loaded}
            onChange={event => {
              const file = event.currentTarget.files?.[0] ?? null
              setWeightsFile(file)
              setError('')
              if (file) {
                setStatus(file.size === PIX2PIX_EDGE2SHOES_WEIGHT_BYTES
                  ? 'Weight file selected. Choose Import to verify and persist it.'
                  : `Selected file is ${formatBytes(file.size)}; expected ${formatBytes(PIX2PIX_EDGE2SHOES_WEIGHT_BYTES)}.`)
              }
            }}
            className="block w-full text-xs text-on-surface-variant file:mr-3 file:rounded-lg file:border file:border-outline file:bg-surface file:px-3 file:py-2 file:text-on-surface"
          />
          <p className="text-xs text-on-surface-muted">
            {weightsFile?.name ?? (installed ? 'Installed copy will be used for Load.' : 'Choose the folded .bin once to install it.')}
          </p>
        </label>

        <label className="space-y-2 text-sm">
          <span className="font-medium">2 · Input image</span>
          <input
            type="file"
            accept="image/*"
            disabled={videoRunning || videoExporting}
            onChange={event => { void chooseInput(event.currentTarget.files?.[0] ?? null) }}
            className="block w-full text-xs text-on-surface-variant file:mr-3 file:rounded-lg file:border file:border-outline file:bg-surface file:px-3 file:py-2 file:text-on-surface"
          />
          <p className="text-xs text-on-surface-muted">Center-cropped and resized locally to 256×256 RGBA8.</p>
        </label>
      </div>

      <div className="rounded-xl border border-outline bg-surface/40 p-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-sm font-semibold">Experimental video flow</p>
            <p className="text-xs text-on-surface-variant">
              Preview samples the first {VIDEO_PREVIEW_SECONDS} seconds at {VIDEO_PREVIEW_FPS} fps. Export processes the full clip at {VIDEO_EXPORT_FPS} fps into a local WebM. Both run frames sequentially through the already-loaded WebGPU model; audio is not preserved yet.
            </p>
          </div>
          <span className="rounded-lg border border-outline px-2 py-1 text-[10px] text-on-surface-muted">
            up to {VIDEO_PREVIEW_FPS * VIDEO_PREVIEW_SECONDS} frames
          </span>
        </div>

        <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="min-w-0 flex-1 space-y-2 text-sm">
            <span className="font-medium">Local video</span>
            <input
              type="file"
              accept="video/*"
              disabled={videoRunning || videoExporting}
              onChange={event => {
                const file = event.currentTarget.files?.[0] ?? null
                setVideoFile(file)
                setVideoStats(null)
                clearVideoExport()
                setError('')
                if (file) {
                  setStatus(loaded
                    ? 'Video selected. Preview or export it with the loaded Pix2Pix model.'
                    : 'Video selected. Load the Pix2Pix model before previewing or exporting.')
                }
              }}
              className="block w-full text-xs text-on-surface-variant file:mr-3 file:rounded-lg file:border file:border-outline file:bg-surface file:px-3 file:py-2 file:text-on-surface"
            />
            <p className="truncate text-xs text-on-surface-muted">{videoFile?.name ?? 'Choose a browser-decodable local video.'}</p>
          </label>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={!loaded || !videoFile || loading || running || videoRunning || videoExporting}
              onClick={() => { void runVideoPreview() }}
              className="rounded-lg bg-secondary px-4 py-2 text-sm font-semibold text-on-secondary disabled:opacity-50"
            >{videoRunning ? 'Processing…' : 'Run video preview'}</button>
            <button
              type="button"
              disabled={!loaded || !videoFile || loading || running || videoRunning || videoExporting}
              onClick={() => { void runVideoExport() }}
              className="rounded-lg border border-secondary px-4 py-2 text-sm font-semibold text-secondary disabled:opacity-50"
            >{videoExporting ? 'Exporting…' : `Export full clip · ${VIDEO_EXPORT_FPS} fps WebM`}</button>
            {videoRunning && (
              <button
                type="button"
                onClick={stopVideoPreview}
                className="rounded-lg border border-outline px-3 py-2 text-sm"
              >Stop after frame</button>
            )}
            {videoExporting && (
              <button
                type="button"
                onClick={stopVideoExport}
                className="rounded-lg border border-outline px-3 py-2 text-sm"
              >Stop export after frame</button>
            )}
          </div>
        </div>

        <video ref={videoElement} className="hidden" muted playsInline preload="metadata" />

        {videoStats && (
          <div className="mt-3 grid gap-2 text-xs text-on-surface-variant sm:grid-cols-4">
            <span><strong className="text-on-surface">{videoStats.frames}</strong> frames</span>
            <span><strong className="text-on-surface">{videoStats.sourceSeconds.toFixed(2)} s</strong> source</span>
            <span><strong className="text-on-surface">{videoStats.effectiveFps.toFixed(1)} fps</strong> effective</span>
            <span><strong className="text-on-surface">{videoStats.averageInferenceMs.toFixed(1)} ms</strong> avg inference</span>
          </div>
        )}

        {videoExportStats && videoExportUrl && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-tertiary/50 bg-tertiary-container/10 p-3 text-xs text-on-surface-variant">
            <div>
              <p className="font-semibold text-on-surface">WebM export ready</p>
              <p className="mt-1">
                {videoExportStats.frames} frames · {videoExportStats.sourceSeconds.toFixed(2)} s · {formatBytes(videoExportStats.bytes)} · {videoExportStats.mimeType}
              </p>
              <p className="mt-1 text-on-surface-muted">
                Video only for now; audio is not preserved in this baseline export.
              </p>
            </div>
            <a
              href={videoExportUrl}
              download={videoExportName}
              className="rounded-lg bg-tertiary px-4 py-2 text-sm font-semibold text-on-tertiary"
            >Download WebM</a>
          </div>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-on-surface-muted">Input</p>
          <canvas ref={inputCanvas} width={256} height={256} className="aspect-square w-full rounded-xl border border-outline bg-surface" />
        </div>
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-on-surface-muted">WebGPU output</p>
          <canvas ref={outputCanvas} width={256} height={256} className="aspect-square w-full rounded-xl border border-outline bg-surface" />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={!weightsFile || checkingStorage || importing || loading || loaded}
          onClick={() => { void importModel() }}
          className="rounded-lg border border-outline px-3 py-2 text-sm font-semibold text-on-surface disabled:opacity-50"
        >{importing ? 'Importing…' : installed ? 'Replace installed model' : 'Import model'}</button>
        <button
          type="button"
          disabled={!installed || checkingStorage || importing || loading || loaded}
          onClick={() => { void loadModel() }}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-on-primary disabled:opacity-50"
        >{loading ? 'Loading…' : 'Load installed model'}</button>
        {loading && (
          <button
            type="button"
            disabled={stopping}
            onClick={() => { void stopLoading() }}
            className="rounded-lg border border-outline px-3 py-2 text-sm disabled:opacity-50"
          >{stopping ? 'Stopping…' : 'Stop loading'}</button>
        )}
        <button
          type="button"
          disabled={!loaded || loading || running || videoRunning || videoExporting}
          onClick={() => { void unload() }}
          className="rounded-lg border border-outline px-3 py-2 text-sm disabled:opacity-50"
        >Unload</button>
        <button
          type="button"
          disabled={!installed || checkingStorage || importing || loading || running || videoRunning || videoExporting || loaded}
          onClick={() => { void removeInstalledModel() }}
          className="rounded-lg border border-error/60 px-3 py-2 text-sm text-error disabled:opacity-50"
        >Remove installed model</button>
        <button
          type="button"
          disabled={!loaded || !input || loading || running || videoRunning || videoExporting}
          onClick={() => { void run() }}
          className="rounded-lg bg-secondary px-4 py-2 text-sm font-semibold text-on-secondary disabled:opacity-50"
        >{running ? 'Running…' : 'Run Pix2Pix'}</button>
      </div>

      <p role="status" className="text-sm text-on-surface-variant">{status}</p>
      {error && <p role="alert" className="rounded-lg border border-error p-3 text-sm text-error">{error}</p>}

      {receipt && (
        <div className="rounded-xl border border-outline p-3 text-xs text-on-surface-variant">
          <p className="font-semibold text-on-surface">Session receipt</p>
          <p className="mt-1">
            WebGPU · load {receipt.loadMs.toFixed(1)} ms · compile {receipt.compileMs.toFixed(1)} ms · inference + readback {receipt.inferenceMs.toFixed(1)} ms
          </p>
          <p className="mt-1">Model: {receipt.modelId} · output 256×256 RGBA8</p>
          <p className="mt-1">This is session evidence; durable qualification remains limited to the pinned model/environment.</p>
        </div>
      )}
    </section>
  )
}
