import type { AudioTranscriber, TranscriptionSegment } from '@mediapipe/tasks-audio'

/** Opt-in browser ASR: model and Wasm are fetched only when the caller loads it. */
export const PARAKEET_MODEL_FILENAME = 'parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm'
export const PARAKEET_MODEL_REVISION = '1f78e5eb6175b1b2cce35f16bcf9e2027b83e8bd'
export const PARAKEET_MODEL_URL = `https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/${PARAKEET_MODEL_REVISION}/${PARAKEET_MODEL_FILENAME}`
export const PARAKEET_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@1.2.2/wasm'

export function validateParakeetAudio(samples: Float32Array): void {
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

/** A stream result contains newly emitted segments, not necessarily a full transcript. */
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
    // A late interim hypothesis must not overwrite finalized speech.
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

/** Caller owns disposal. No remote ASR or hidden provider fallback. */
export async function createParakeetTranscriber(localModel?: File): Promise<AudioTranscriber> {
  if (localModel) validateParakeetModelFile(localModel)
  const { AudioTranscriber, FilesetResolver } = await import('@mediapipe/tasks-audio')
  const fileset = await FilesetResolver.forAudioTasks(PARAKEET_WASM_URL)
  const baseOptions = localModel
    ? { modelAssetBuffer: new Uint8Array(await localModel.arrayBuffer()), delegate: 'CPU' as const }
    : { modelAssetPath: PARAKEET_MODEL_URL, delegate: 'CPU' as const }
  const transcriber = await AudioTranscriber.createFromOptions(fileset, {
    baseOptions,
    sampleRateHz: 16_000,
    enableInterimResults: true,
  })
  return transcriber
}
