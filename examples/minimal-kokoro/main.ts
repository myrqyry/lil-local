import { createHttpAssetResolver, type RuntimeContext } from '@lil-local/inference-core'
import { KokoroPipeline } from '@lil-local/kokoro'
import './app.css'

const status = document.querySelector<HTMLElement>('#status')!
const error = document.querySelector<HTMLElement>('#error')!
const button = document.querySelector<HTMLButtonElement>('#synthesize')!
const text = document.querySelector<HTMLTextAreaElement>('#phrase')!
const receipt = document.querySelector<HTMLElement>('#receipt')!
const receiptValues = document.querySelector<HTMLElement>('#receipt-values')!
const pipeline = new KokoroPipeline()

function showReceipt(result: Awaited<ReturnType<KokoroPipeline['run']>>): void {
  const values = {
    backend: result.receipt.backend,
    audio: `${result.durationSeconds.toFixed(3)}s, ${result.samples.length} samples`,
    sampleRate: `${result.sampleRate}Hz`,
    load: `${result.receipt.loadMs.toFixed(0)}ms`,
    inference: `${result.receipt.inferenceMs.toFixed(0)}ms`,
    warnings: result.receipt.warnings.join(', ') || 'none',
  }
  receiptValues.innerHTML = Object.entries(values).map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('')
  receipt.hidden = false
}

async function play(samples: Float32Array, sampleRate: number): Promise<void> {
  const context = new AudioContext()
  const buffer = context.createBuffer(1, samples.length, sampleRate)
  buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0)
  const source = context.createBufferSource()
  source.buffer = buffer
  source.connect(context.destination)
  source.addEventListener('ended', () => void context.close(), { once: true })
  source.start()
}

async function load(): Promise<void> {
  try {
    const assets = createHttpAssetResolver('/models/kokoro/')
    const context: RuntimeContext = {
      backend: 'wasm',
      assets,
      liteRt: {
        loadModel: async () => { throw new Error('Kokoro does not use LiteRT models') },
        loadNpy: async () => { throw new Error('Kokoro does not use NPY assets') },
        fetchBuffer: async () => { throw new Error('Kokoro does not use LiteRT buffers') },
      },
    }
    await pipeline.load(context)
    status.textContent = 'Ready (wasm)'
    button.disabled = false
  } catch (cause) {
    status.textContent = 'Load failed'
    error.textContent = cause instanceof Error ? cause.message : String(cause)
    error.hidden = false
  }
}

button.disabled = true
button.addEventListener('click', async () => {
  button.disabled = true
  error.hidden = true
  status.textContent = 'Synthesizing...'
  try {
    const result = await pipeline.run({ text: text.value })
    showReceipt(result)
    await play(result.samples, result.sampleRate)
    status.textContent = 'Played'
  } catch (cause) {
    status.textContent = 'Inference failed'
    error.textContent = cause instanceof Error ? cause.message : String(cause)
    error.hidden = false
  } finally {
    button.disabled = false
  }
})

void load()
