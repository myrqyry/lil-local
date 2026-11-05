export const MEDIA_PIPE_AUDIO_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@1.2.2/wasm'

export function audioAbortError(): Error {
  const error = new Error('MediaPipe Audio load aborted')
  error.name = 'AbortError'
  return error
}

export function checkAudioAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw audioAbortError()
}

export async function resolveAudioFileset(wasmBase = MEDIA_PIPE_AUDIO_WASM_BASE) {
  if (!wasmBase.trim()) throw new Error('MediaPipe Audio wasmBase must be nonempty')
  const { FilesetResolver } = await import('@mediapipe/tasks-audio')
  return FilesetResolver.forAudioTasks(wasmBase)
}
