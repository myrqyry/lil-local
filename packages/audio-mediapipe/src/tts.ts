import type { BrowserSubstrate } from '@lil-local/inference-core'
import type { TextToSpeech, TextToSpeechResult } from '@mediapipe/tasks-audio'
import { MEDIA_PIPE_AUDIO_WASM_BASE, checkAudioAbort, resolveAudioFileset } from './shared'

export type MediaPipeTtsOptions = {
  /** Explicit .litertlm model URL, local bytes, or Blob. No unqualified default model. */
  model: string | Blob | Uint8Array
  wasmBase?: string
  delegate?: 'CPU' | 'GPU'
  defaultVoiceId?: string
  defaultLocale?: string
  speakingRate?: number
}
export type MediaPipeTtsInput = { text: string; voiceId?: string; locale?: string }
export type MediaPipeTtsResult = {
  kind: 'audio'
  samples: Float32Array
  sampleRateHz: number
  channels: 1
  wordTimestamps?: TextToSpeechResult['wordTimestamps']
}

export function validateTtsText(text: string): string {
  if (typeof text !== 'string' || !text.trim()) throw new Error('TTS text cannot be empty')
  return text.trim()
}
function validateTtsOptions(options: MediaPipeTtsOptions): void {
  if (!options?.model || (typeof options.model === 'string' && !options.model.trim()))
    throw new Error('MediaPipe TTS requires an explicit model')
  if (options.model instanceof Uint8Array && !options.model.byteLength)
    throw new Error('MediaPipe TTS model is empty')
  if (options.model instanceof Blob && !options.model.size)
    throw new Error('MediaPipe TTS model is empty')
  if (options.speakingRate !== undefined &&
      (!Number.isFinite(options.speakingRate) || options.speakingRate <= 0 || options.speakingRate > 4))
    throw new Error('MediaPipe TTS speakingRate must be greater than 0 and at most 4')
}
export function validateTtsResult(output: TextToSpeechResult): MediaPipeTtsResult {
  if (!(output.samples instanceof Float32Array) || !output.samples.length ||
      !Number.isFinite(output.sampleRateHz) || output.sampleRateHz <= 0)
    throw new Error('MediaPipe TTS returned invalid PCM audio')
  for (const sample of output.samples) {
    if (!Number.isFinite(sample)) throw new Error('MediaPipe TTS returned non-finite PCM')
    if (sample < -1 || sample > 1) throw new Error('MediaPipe TTS returned non-normalized PCM')
  }
  return {
    kind: 'audio',
    samples: output.samples,
    sampleRateHz: output.sampleRateHz,
    channels: 1,
    wordTimestamps: output.wordTimestamps,
  }
}

/**
 * Real MediaPipe speech engine, with listVoices()/synthesizeStream() available
 * to downstream consumers. Requires explicit model, no auto model selection.
 * The caller owns close(); abort is checked across SDK initialization.
 */
export async function createMediaPipeTtsEngine(
  options: MediaPipeTtsOptions,
  signal?: AbortSignal,
): Promise<TextToSpeech> {
  validateTtsOptions(options)
  checkAudioAbort(signal)
  const { TextToSpeech } = await import('@mediapipe/tasks-audio')
  checkAudioAbort(signal)
  const fileset = await resolveAudioFileset(options.wasmBase ?? MEDIA_PIPE_AUDIO_WASM_BASE)
  checkAudioAbort(signal)
  const baseOptions = typeof options.model === 'string'
    ? {modelAssetPath: options.model, delegate: options.delegate ?? 'CPU'}
    : {modelAssetBuffer: options.model instanceof Uint8Array
      ? options.model : new Uint8Array(await options.model.arrayBuffer()),
      delegate: options.delegate ?? 'CPU'}
  checkAudioAbort(signal)
  const engine = await TextToSpeech.createFromOptions(fileset, {
    baseOptions,
    defaultVoiceId: options.defaultVoiceId,
    defaultLocale: options.defaultLocale,
    speakingRate: options.speakingRate,
    cacheDirectory: ':nocache',
  })
  if (signal?.aborted) {
    engine.close()
    checkAudioAbort(signal)
  }
  return engine
}

export function createMediaPipeTtsSubstrate(
  options: MediaPipeTtsOptions,
): BrowserSubstrate<MediaPipeTtsInput, MediaPipeTtsResult> {
  validateTtsOptions(options)
  return {
    id: 'mediapipe:tts-litertlm',
    format: 'litertlm',
    capabilities: ['text-to-speech'],
    requestedBackends: [options.delegate === 'GPU' ? 'webgpu' : 'wasm'],
    async create(_progress, signal) {
      const engine = await createMediaPipeTtsEngine(options, signal)
      return {
        async infer({text, voiceId, locale}) {
          const output = await engine.synthesize(validateTtsText(text),{voiceId,locale})
          return validateTtsResult(output)
        },
        async dispose() {
          try { engine.cancel() } finally { engine.close() }
        },
      }
    },
  }
}
