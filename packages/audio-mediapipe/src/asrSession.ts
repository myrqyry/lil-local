import type { AudioTranscriber, AudioTranscriberResult, TranscriptionSegment } from '@mediapipe/tasks-audio'
import { combineParakeetSegments, parakeetTranscript, validateParakeetAudio } from './parakeetMediaPipe'

export interface MediaPipeAsrTranscript {
  text: string
  segments: TranscriptionSegment[]
}

export interface MediaPipeAsrStreamSessionOptions {
  /** Keep a caller-owned engine open when the session is released. */
  closeEngineOnClose?: boolean
}

export interface MediaPipeAsrStreamSession {
  /** Enqueue a chunk of normalized 16 kHz mono PCM, in recording order. */
  push(samples: Float32Array): Promise<MediaPipeAsrTranscript>
  /** Drain all queued audio before flushing the native stream. */
  finish(): Promise<MediaPipeAsrTranscript>
  /** Invalidate pending deliveries; optionally close the supplied engine after native work settles. */
  close(): Promise<void>
}

function closedError(): Error {
  const error = new Error('ASR session is closed')
  error.name = 'AbortError'
  return error
}

/**
 * Caller-provided MediaPipe engine; closing the session closes the engine by
 * default unless the caller explicitly retains ownership.
 * Capture, resampling, permission, audio clock and model loading stay with the
 * consumer. Native inference may not be interruptible once underway.
 */
export function createMediaPipeAsrStreamSession(
  engine: AudioTranscriber,
  options: MediaPipeAsrStreamSessionOptions = {},
): MediaPipeAsrStreamSession {
  let closed = false
  let finishing = false
  let failure: unknown
  let pending: Promise<void> = Promise.resolve()
  let finishPromise: Promise<MediaPipeAsrTranscript> | undefined
  let closePromise: Promise<void> | undefined
  let segments: TranscriptionSegment[] = []
  let text = ''

  const snapshot = (): MediaPipeAsrTranscript => ({
    text,
    segments: segments.map(segment => ({ ...segment })),
  })

  const accept = (result: AudioTranscriberResult): MediaPipeAsrTranscript => {
    if (result.segments?.length) {
      segments = combineParakeetSegments(
        segments,
        result.segments.map(segment => ({ ...segment })),
      )
      text = parakeetTranscript(segments)
    } else if (result.text) {
      text = result.text
    }
    return snapshot()
  }

  const enqueue = <T>(operation: () => Promise<T>): Promise<T> => {
    const task = pending.then(async () => {
      if (closed) throw closedError()
      if (failure !== undefined) throw failure
      try {
        const result = await operation()
        if (closed) throw closedError()
        return result
      } catch (error) {
        if (closed) throw closedError()
        failure = error
        throw error
      }
    })
    pending = task.then(() => undefined, () => undefined)
    return task
  }

  return {
    async push(samples) {
      if (closed) throw closedError()
      if (finishing) throw new Error('ASR session is finishing or finished')
      validateParakeetAudio(samples)
      // AudioWorklet buffers may be reused after their message arrives.
      const owned = new Float32Array(samples)
      return enqueue(async () => accept(await engine.transcribeStream(owned)))
    },
    finish() {
      if (closed) return Promise.reject(closedError())
      if (finishPromise) return finishPromise
      finishing = true
      finishPromise = enqueue(async () => accept(await engine.finishStream()))
      return finishPromise
    },
    close() {
      if (closePromise) return closePromise
      closed = true
      // Don't close native WASM resources while an in-flight inference still runs.
      closePromise = pending.then(() => {
        if (options.closeEngineOnClose !== false) engine.close()
      })
      return closePromise
    },
  }
}
