export interface FrameSequenceProgress {
  completed: number
  total: number
  time: number
}

export function buildPreviewFrameTimes(
  durationSeconds: number,
  fps: number,
  maxSeconds: number,
): number[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return []
  if (!Number.isFinite(fps) || fps <= 0) throw new Error('fps must be positive')
  if (!Number.isFinite(maxSeconds) || maxSeconds <= 0) throw new Error('maxSeconds must be positive')

  const end = Math.min(durationSeconds, maxSeconds)
  const interval = 1 / fps
  const times: number[] = []
  for (let time = 0; time < end; time += interval) {
    times.push(Number(time.toFixed(6)))
  }
  return times
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
