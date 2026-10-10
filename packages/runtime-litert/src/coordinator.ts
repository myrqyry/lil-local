export type InferenceEvent = 'queued' | 'started' | 'finished'

export interface InferenceEventDetails {
  label: string
  queueDepth: number
  queuedMs?: number
  durationMs?: number
}

export interface InferenceCoordinatorSnapshot {
  active: boolean
  activeLabel?: string
  queued: number
}

export interface InferenceRunOptions {
  /** Higher priority runs first among waiting jobs; omitted priority preserves FIFO. */
  priority?: number
}

interface QueuedTask {
  priority: number
  start: (remainingQueued: number) => void
}

type InferenceListener = (details: InferenceEventDetails) => void

function abortError(): Error {
  if (typeof DOMException !== 'undefined') return new DOMException('Inference was aborted', 'AbortError')
  const error = new Error('Inference was aborted')
  error.name = 'AbortError'
  return error
}

export class InferenceCoordinator {
  private activeLabel: string | undefined
  private queue: QueuedTask[] = []
  private scheduled = false
  private listeners = new Map<InferenceEvent, Set<InferenceListener>>()

  isBusy(): boolean {
    return this.activeLabel !== undefined
  }

  snapshot(): InferenceCoordinatorSnapshot {
    return { active: this.isBusy(), activeLabel: this.activeLabel, queued: this.queue.length }
  }

  on(event: InferenceEvent, listener: InferenceListener): () => void {
    const listeners = this.listeners.get(event) ?? new Set<InferenceListener>()
    listeners.add(listener)
    this.listeners.set(event, listeners)
    return () => listeners.delete(listener)
  }

  run<T>(
    task: () => Promise<T>,
    signal?: AbortSignal,
    label = 'inference',
    options: InferenceRunOptions = {},
  ): Promise<T> {
    if (signal?.aborted) return Promise.reject(abortError())
    const priority = options.priority ?? 0
    if (!Number.isFinite(priority)) {
      return Promise.reject(new RangeError('Inference priority must be finite'))
    }

    const queuedAt = performance.now()
    return new Promise<T>((resolve, reject) => {
      let queued!: QueuedTask
      const onAbort = () => {
        const index = this.queue.indexOf(queued)
        if (index < 0) return // Already running: caller/task owns in-flight cancellation.
        this.queue.splice(index, 1)
        signal?.removeEventListener('abort', onAbort)
        reject(abortError())
      }

      queued = {
        priority,
        start: (remainingQueued) => {
          signal?.removeEventListener('abort', onAbort)
          if (signal?.aborted) {
            reject(abortError())
            this.schedule()
            return
          }

          this.activeLabel = label
          const startedAt = performance.now()
          this.emit('started', {
            label,
            queueDepth: remainingQueued,
            queuedMs: startedAt - queuedAt,
          })

          // A running LiteRT invocation is not preemptible by a later priority job.
          void Promise.resolve().then(task).then(resolve, reject).finally(() => {
            const durationMs = performance.now() - startedAt
            this.activeLabel = undefined
            this.emit('finished', { label, queueDepth: this.queue.length, durationMs })
            this.schedule()
          })
        },
      }

      this.queue.push(queued)
      signal?.addEventListener('abort', onAbort, { once: true })
      this.emit('queued', { label, queueDepth: this.queue.length })
      // A queued listener may have cancelled the job while events were emitted.
      if (signal?.aborted) onAbort()
      this.schedule()
    })
  }

  private schedule(): void {
    if (this.isBusy() || this.scheduled) return
    this.scheduled = true
    queueMicrotask(() => {
      this.scheduled = false
      if (this.isBusy() || this.queue.length === 0) return

      let next = 0
      for (let i = 1; i < this.queue.length; i += 1) {
        // Strict > preserves FIFO order for equal-priority jobs.
        if (this.queue[i]!.priority > this.queue[next]!.priority) next = i
      }
      const [job] = this.queue.splice(next, 1)
      job!.start(this.queue.length)
    })
  }

  // ponytail: one bad listener must not poison the inference it observes.
  private emit(event: InferenceEvent, details: InferenceEventDetails): void {
    this.listeners.get(event)?.forEach((listener) => {
      try {
        listener(details)
      } catch (error) {
        console.error(`inference ${event} listener failed`, error)
      }
    })
  }
}

export const inferenceCoordinator = new InferenceCoordinator()
