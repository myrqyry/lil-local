import type { BrowserSubstrate } from '@lil-local/inference-core'
import type { AudioTranscriber, AudioTranscriberResult, TranscriptionSegment } from '@mediapipe/tasks-audio'
import { MEDIA_PIPE_AUDIO_WASM_BASE, checkAudioAbort, resolveAudioFileset } from './shared'

/** Opt-in on-device ASR; no weights or WASM are loaded during discovery. */
export const PARAKEET_MODEL_FILENAME = 'parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm'
export const PARAKEET_MODEL_REVISION = '1f78e5eb6175b1b2cce35f16bcf9e2027b83e8bd'
export const PARAKEET_MODEL_URL =
  `https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/${PARAKEET_MODEL_REVISION}/${PARAKEET_MODEL_FILENAME}`
export const PARAKEET_WASM_URL = MEDIA_PIPE_AUDIO_WASM_BASE

/** Publisher's LFS provenance (not yet verified against a browser-downloaded model). */
export const PARAKEET_MODEL_ASSET = {
  url: PARAKEET_MODEL_URL,
  bytes: 614_588_752,
  sha256: '66c5a74cf6c2c1d2b990901f6cece5bb53e3214dbceafa0989900a6f0eeb3504',
} as const

export function validateParakeetAudio(samples: Float32Array): void {
  if (!(samples instanceof Float32Array)) throw new Error('Audio must contain Float32 PCM samples')
  if (samples.length === 0) throw new Error('Audio is empty')
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new Error('Audio must contain finite PCM samples')
    if (sample < -1 || sample > 1) throw new Error('Audio must contain normalized PCM [-1, 1]')
  }
}

export function validateParakeetModelFile(file: Pick<File, 'name' | 'size'>): void {
  if (!file.name.toLowerCase().endsWith('.litertlm')) throw new Error('Model must be a .litertlm file')
  if (file.size <= 0) throw new Error('Model file is empty')
}

/** Reconcile time-overlapping interim and finalized segments across stream chunks. */
export function combineParakeetSegments(
  previous: readonly TranscriptionSegment[],
  incoming: readonly TranscriptionSegment[],
): TranscriptionSegment[] {
  let merged = [...previous]
  for (const segment of incoming) {
    if (!segment.transcript.trim()) continue
    merged = merged.filter(existing =>
      existing.toTimestampMs <= segment.fromTimestampMs
      || existing.fromTimestampMs >= segment.toTimestampMs
      || (existing.isFinal && !segment.isFinal)
    )
    if (segment.isFinal || !merged.some(existing =>
      existing.isFinal
      && existing.fromTimestampMs < segment.toTimestampMs
      && segment.fromTimestampMs < existing.toTimestampMs
    )) merged.push(segment)
  }
  return merged.sort((a, b) => a.fromTimestampMs - b.fromTimestampMs)
}

export function parakeetTranscript(segments: readonly TranscriptionSegment[]): string {
  return segments.map(segment => segment.transcript.trim()).filter(Boolean).join(' ').trim()
}

export type ParakeetOptions = {
  /** Publisher-matching local LiteRT-LM bundle, or omit for pinned remote model URL. */
  model?: Blob | Uint8Array
  wasmBase?: string
}

export interface ParakeetLoadOptions {
  wasmBase?: string
  signal?: AbortSignal
}

/**
 * Exposes the SDK's real transcribeStream/finishStream/prewarm methods so
 * downstream apps can own microphone/worklet scheduling. Caller owns close().
 *
 * Remote model URLs are revision-pinned but NOT integrity-checked by the SDK.
 * This is not advertised as checksum-verified until browser runtime evidence exists.
 */
export async function createParakeetTranscriber(
  localModel?: Blob | Uint8Array,
  options: ParakeetLoadOptions = {},
): Promise<AudioTranscriber> {
  checkAudioAbort(options.signal)
  if (localModel instanceof Blob) {
    if ('name' in localModel && typeof localModel.name === 'string')
      validateParakeetModelFile(localModel as File)
    if (localModel.size <= 0) throw new Error('Model file is empty')
  }
  if (localModel instanceof Uint8Array && localModel.byteLength === 0)
    throw new Error('Model file is empty')
  const { AudioTranscriber } = await import('@mediapipe/tasks-audio')
  checkAudioAbort(options.signal)
  const fileset = await resolveAudioFileset(options.wasmBase ?? PARAKEET_WASM_URL)
  checkAudioAbort(options.signal)
  const baseOptions = localModel
    ? { modelAssetBuffer: localModel instanceof Uint8Array
      ? localModel : new Uint8Array(await localModel.arrayBuffer()), delegate: 'CPU' as const }
    : { modelAssetPath: PARAKEET_MODEL_URL, delegate: 'CPU' as const }
  checkAudioAbort(options.signal)
  const transcriber = await AudioTranscriber.createFromOptions(fileset, {
    baseOptions,
    sampleRateHz: 16_000,
    enableInterimResults: true,
    cacheDir: ':nocache',
  })
  if (options.signal?.aborted) {
    transcriber.close()
    checkAudioAbort(options.signal)
  }
  return transcriber
}

/** Unary Parakeet for substrate consumers; use createParakeetTranscriber for streaming. */
export function createParakeetSubstrate(
  options: ParakeetOptions = {},
): BrowserSubstrate<Float32Array, AudioTranscriberResult> {
  return {
    id: 'mediapipe:parakeet-tdt-0.6b-v3',
    format: 'litertlm',
    capabilities: ['speech-to-text'],
    requestedBackends: ['wasm'],
    async create(_progress, signal) {
      const engine = await createParakeetTranscriber(options.model, {
        wasmBase: options.wasmBase,
        signal,
      })
      try {
        await engine.prewarm()
        checkAudioAbort(signal)
      } catch (error) {
        engine.close()
        throw error
      }
      return {
        async infer(samples) {
          validateParakeetAudio(samples)
          return engine.transcribe(samples, true)
        },
        async dispose() { engine.close() },
      }
    },
  }
}
