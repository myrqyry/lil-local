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

type Runner = BrowserSubstrateRunner<ImageInput, Pix2PixImageResult, Pix2PixLoadProgress>

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
  const [stopping, setStopping] = useState(false)
  const [receipt, setReceipt] = useState<InferenceReceipt | null>(null)

  const inputCanvas = useRef<HTMLCanvasElement | null>(null)
  const outputCanvas = useRef<HTMLCanvasElement | null>(null)
  const runner = useRef<Runner | null>(null)
  const pendingRunner = useRef<Runner | null>(null)
  const loadAbort = useRef<AbortController | null>(null)
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
      const pending = pendingRunner.current
      pendingRunner.current = null
      loadAbort.current = null
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
    if (!installed || loading || running || importing || runner.current) return
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
    if (loading || running) return
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
    if (!active || !input || loading || running) return
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
            onChange={event => { void chooseInput(event.currentTarget.files?.[0] ?? null) }}
            className="block w-full text-xs text-on-surface-variant file:mr-3 file:rounded-lg file:border file:border-outline file:bg-surface file:px-3 file:py-2 file:text-on-surface"
          />
          <p className="text-xs text-on-surface-muted">Center-cropped and resized locally to 256×256 RGBA8.</p>
        </label>
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
          disabled={!loaded || loading || running}
          onClick={() => { void unload() }}
          className="rounded-lg border border-outline px-3 py-2 text-sm disabled:opacity-50"
        >Unload</button>
        <button
          type="button"
          disabled={!installed || checkingStorage || importing || loading || running || loaded}
          onClick={() => { void removeInstalledModel() }}
          className="rounded-lg border border-error/60 px-3 py-2 text-sm text-error disabled:opacity-50"
        >Remove installed model</button>
        <button
          type="button"
          disabled={!loaded || !input || loading || running}
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
