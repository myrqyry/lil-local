import type { BrowserSubstrate } from '@lil-local/inference-core'
import type { AudioClassifierResult } from '@mediapipe/tasks-audio'
import { MEDIA_PIPE_AUDIO_WASM_BASE, checkAudioAbort, resolveAudioFileset } from './shared'

export const YAMNET_MODEL = {
  url: 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite',
  bytes: 4_126_810,
  sha256: '4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf',
} as const

export type AudioClip = { samples: Float32Array; sampleRate: number }
export type AudioClassificationResult = {
  kind: 'audio-classification'
  windows: {
    timestampMs: number | null
    heads: { headIndex: number; headName: string; categories: { categoryName: string; displayName: string; index: number; score: number }[] }[]
  }[]
}
export type AudioLoadProgress = 'model' | 'integrity' | 'wasm'
export type YamNetOptions = {
  /** Verified official model bytes. No model fetch if this is provided. */
  model?: Blob | Uint8Array
  /** Same-version MediaPipe audio WASM URL. */
  wasmBase?: string
  maxResults?: number
  scoreThreshold?: number
}

export function validateAudioClip(clip: AudioClip): void {
  if (!clip || !(clip.samples instanceof Float32Array)) throw new Error('Audio samples must be Float32Array')
  if (!Number.isFinite(clip.sampleRate) || clip.sampleRate < 1)
    throw new Error('Audio sample rate must be positive')
  if (!clip.samples.length) throw new Error('Audio is empty')
  for (const value of clip.samples) {
    if (!Number.isFinite(value)) throw new Error('Audio must contain finite PCM samples')
    if (value < -1 || value > 1) throw new Error('Audio must contain normalized PCM [-1, 1]')
  }
}

export function mapYamNetResults(results: readonly AudioClassifierResult[]): AudioClassificationResult {
  return {
    kind: 'audio-classification',
    windows: results.map(result => ({
      timestampMs: result.timestampMs ?? null,
      heads: result.classifications.map(head => ({
        headIndex: head.headIndex,
        headName: head.headName,
        categories: head.categories.map(category => ({
          categoryName: category.categoryName,
          displayName: category.displayName,
          index: category.index,
          score: category.score,
        })),
      })),
    })),
  }
}

export async function verifyYamNetModelBytes(
  bytes: Uint8Array,
  expected: { bytes: number; sha256: string } = YAMNET_MODEL,
): Promise<void> {
  if (bytes.byteLength !== expected.bytes) throw new Error('YAMNet model size mismatch')
  if (!globalThis.crypto?.subtle)
    throw new Error('YAMNet SHA-256 requires a secure WebCrypto context')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', Uint8Array.from(bytes))
  const actual = [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2,'0')).join('')
  if (actual !== expected.sha256) throw new Error('YAMNet model SHA-256 mismatch')
}

async function verifiedModelBytes(
  model: Blob | Uint8Array | undefined,
  progress?: (value: AudioLoadProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  checkAudioAbort(signal)
  progress?.('model')
  let bytes: Uint8Array
  if (model instanceof Uint8Array) {
    bytes = model
  } else if (model) {
    bytes = new Uint8Array(await model.arrayBuffer())
  } else {
    const response = await fetch(YAMNET_MODEL.url, { signal })
    if (!response.ok) throw new Error('YAMNet model fetch failed: HTTP ' + response.status)
    const length = response.headers.get('content-length')
    if (length && Number(length) !== YAMNET_MODEL.bytes)
      throw new Error('YAMNet model size mismatch in HTTP response')
    bytes = new Uint8Array(await response.arrayBuffer())
  }
  checkAudioAbort(signal)
  progress?.('integrity')
  await verifyYamNetModelBytes(bytes)
  checkAudioAbort(signal)
  return bytes
}

/** Public, opt-in sound-event classifier; no microphone capture or upload. */
export function createYamNetSubstrate(
  options: YamNetOptions = {},
): BrowserSubstrate<AudioClip, AudioClassificationResult, AudioLoadProgress> {
  if (options.maxResults !== undefined && (!Number.isInteger(options.maxResults) || (options.maxResults < 1 && options.maxResults !== -1)))
    throw new Error('maxResults must be -1 or a positive integer')
  if (options.scoreThreshold !== undefined && (!Number.isFinite(options.scoreThreshold) || options.scoreThreshold < 0 || options.scoreThreshold > 1))
    throw new Error('scoreThreshold must be between 0 and 1')
  return {
    id: 'mediapipe:yamnet-float32-v1',
    format: 'tflite',
    capabilities: ['audio-classification'],
    requestedBackends: ['wasm'],
    async create(progress, signal) {
      const model = await verifiedModelBytes(options.model, progress, signal)
      progress?.('wasm')
      const { AudioClassifier } = await import('@mediapipe/tasks-audio')
      checkAudioAbort(signal)
      const fileset = await resolveAudioFileset(options.wasmBase ?? MEDIA_PIPE_AUDIO_WASM_BASE)
      checkAudioAbort(signal)
      const classifier = await AudioClassifier.createFromOptions(fileset, {
        baseOptions: {modelAssetBuffer: model, delegate: 'CPU'},
        maxResults: options.maxResults,
        scoreThreshold: options.scoreThreshold,
      })
      if (signal?.aborted) {
        classifier.close()
        checkAudioAbort(signal)
      }
      return {
        async infer(clip) {
          validateAudioClip(clip)
          return mapYamNetResults(classifier.classify(clip.samples,clip.sampleRate))
        },
        async dispose() { classifier.close() },
      }
    },
  }
}
