export const MAX_VIDEO_EXPORT_SECONDS = 60

export function frameDurationSeconds(time: number, fps: number, duration: number): number {
  return Math.max(0, Math.min(1 / fps, duration - time))
}

export interface FrameSequenceProgress {
  completed: number
  total: number
  time: number
}

export function buildFrameTimes(
  durationSeconds: number,
  fps: number,
): number[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return []
  if (!Number.isFinite(fps) || fps <= 0) throw new Error('fps must be positive')
  if (durationSeconds > MAX_VIDEO_EXPORT_SECONDS) throw new Error('Video exceeds maximum 60-second export duration')

  const interval = 1 / fps
  const times: number[] = []
  for (let time = 0; time < durationSeconds; time += interval) {
    times.push(Number(time.toFixed(6)))
  }
  return times
}

export function buildPreviewFrameTimes(
  durationSeconds: number,
  fps: number,
  maxSeconds: number,
): number[] {
  if (!Number.isFinite(maxSeconds) || maxSeconds <= 0) throw new Error('maxSeconds must be positive')
  return buildFrameTimes(Math.min(durationSeconds, maxSeconds), fps)
}

export function chooseVideoExportMimeType(
  isTypeSupported: (mimeType: string) => boolean,
): string {
  for (const mimeType of ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']) {
    if (isTypeSupported(mimeType)) return mimeType
  }
  throw new Error('WebM recording is unavailable in this browser')
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException('Frame sequence aborted', 'AbortError')
}

export async function runFrameSequence(options: {
  times: readonly number[]
  signal: AbortSignal
  processFrame: (time: number, index: number) => Promise<void>
  onProgress?: (progress: FrameSequenceProgress) => void
}): Promise<number> {
  const { times, signal, processFrame, onProgress } = options
  throwIfAborted(signal)

  let completed = 0
  for (let index = 0; index < times.length; index += 1) {
    throwIfAborted(signal)
    const time = times[index]
    await processFrame(time, index)
    completed += 1
    onProgress?.({ completed, total: times.length, time })
    throwIfAborted(signal)
  }
  return completed
}
