import type { TextToSpeech, TextToSpeechStream, VoiceInfo } from '@mediapipe/tasks-audio'
import { validateTtsResult, validateTtsText, type MediaPipeTtsResult } from './tts'

/** Playback-neutral PCM chunks: consumers own resampling, queuing and output. */
export type MediaPipeTtsStreamChunk = MediaPipeTtsResult & {
  chunkIndex: number
  isFinal: boolean
}

export type MediaPipeTtsUtteranceOptions = {
  voiceId?: string
  locale?: string
  signal?: AbortSignal
}

export type MediaPipeTtsStreamSession = {
  listVoices(): Promise<VoiceInfo[]>
  stream(text: string, options?: MediaPipeTtsUtteranceOptions): AsyncIterable<MediaPipeTtsStreamChunk>
  cancel(): void
  close(): void
}

function abortError(): Error {
  const error = new Error('TTS synthesis aborted')
  error.name = 'AbortError'
  return error
}

/**
 * Consumer-facing adapter around a caller-loaded MediaPipe engine.
 *
 * A session allows exactly one utterance at a time. Cancellation invalidates
 * delivery even if the native read resolves late; it cannot guarantee that
 * native GPU/WASM computation stops immediately. No playback or worker is
 * created here. The session owns the engine and closes it on close().
 */
export function createMediaPipeTtsStreamSession(engine: TextToSpeech): MediaPipeTtsStreamSession {
  let closed = false
  let active: { cancelled: boolean; native?: TextToSpeechStream } | undefined

  const cancel = () => {
    if (!active || active.cancelled) return
    active.cancelled = true
    try { active.native?.cancel() } finally { engine.cancel() }
  }

  return {
    listVoices() {
      if (closed) return Promise.reject(new Error('TTS session is closed'))
      return engine.listVoices()
    },
    async *stream(text, options = {}) {
      if (closed) throw new Error('TTS session is closed')
      if (active) throw new Error('TTS session already active')
      const normalized = validateTtsText(text)
      const current: { cancelled: boolean; native?: TextToSpeechStream } = { cancelled: false }
      let completed = false
      active = current
      const onAbort = () => cancel()
      options.signal?.addEventListener('abort', onAbort, { once: true })

      try {
        if (options.signal?.aborted) cancel()
        if (current.cancelled) throw abortError()

        current.native = await engine.synthesizeStream(normalized, {
          voiceId: options.voiceId,
          locale: options.locale,
        })
        if (current.cancelled) throw abortError()

        let lastIndex = -1
        while (true) {
          let chunk: Awaited<ReturnType<TextToSpeechStream['readNext']>>
          try {
            chunk = await current.native.readNext()
          } catch (error) {
            if (current.cancelled) throw abortError()
            throw error
          }
          if (current.cancelled) throw abortError()
          if (chunk === null) {
            completed = true
            break
          }
          if (!Number.isSafeInteger(chunk.chunkIndex) || chunk.chunkIndex <= lastIndex)
            throw new Error('MediaPipe TTS returned out-of-order PCM chunks')
          lastIndex = chunk.chunkIndex
          const pcm = validateTtsResult(chunk)
          // Native samples can be views into the reusable WASM heap.
          yield { ...pcm, samples: new Float32Array(pcm.samples), chunkIndex: chunk.chunkIndex, isFinal: chunk.isFinal }
          if (chunk.isFinal) {
            completed = true
            break
          }
        }
      } finally {
        options.signal?.removeEventListener('abort', onAbort)
        // A consumer can stop iterating before the native stream finishes.
        // Always stop its compute; closing resources alone is not cancellation.
        if (!completed && !current.cancelled) cancel()
        try {
          current.native?.close()
        } finally {
          if (active === current) active = undefined
        }
      }
    },
    cancel,
    close() {
      if (closed) return
      closed = true
      try { cancel() } finally { engine.close() }
    },
  }
}
