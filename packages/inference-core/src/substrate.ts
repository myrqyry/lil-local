import type { Backend, Capability } from './types'

/**
 * A browser runtime family, not a GPU execution provider.
 * A substrate may report requested execution backends, but its model-specific
 * results are responsible for any claims about what actually executed.
 */
export interface BrowserSubstrate<I, O, Progress = never> {
  readonly id: string
  readonly format: string
  readonly capabilities: readonly Capability[]
  readonly requestedBackends: readonly Backend[]
  /** Called only by explicit load(); discovery never imports weights or starts inference. */
  create(onProgress?: (progress: Progress) => void, signal?: AbortSignal): Promise<LoadedBrowserSubstrate<I, O>>
}

export interface LoadedBrowserSubstrate<I, O> {
  infer(input: I): Promise<O>
  dispose(): Promise<void>
}

export type SubstrateStatus = 'idle' | 'loading' | 'ready' | 'running' | 'failed' | 'disposing' | 'disposed'

function loadAborted(): Error {
  const error = new Error('Substrate load aborted')
  error.name = 'AbortError'
  return error
}

/**
 * Owns exactly one substrate/model session. No implicit model downloads,
 * backend fallback, preemption, durable evidence promotion, or cross-runtime
 * GPU-memory budgeting. Parallel models use separate runners.
 */
export class BrowserSubstrateRunner<I, O, Progress = never> {
  private instance: LoadedBrowserSubstrate<I, O> | null = null
  private loading: Promise<void> | null = null
  private running: Promise<O> | null = null
  private disposing: Promise<void> | null = null
  private currentStatus: SubstrateStatus = 'idle'
  private lateCleanupFailure: { error: unknown } | null = null

  constructor(readonly substrate: BrowserSubstrate<I, O, Progress>) {
    if (!substrate.id.trim() || !substrate.format.trim() ||
        substrate.capabilities.length === 0 || substrate.requestedBackends.length === 0) {
      throw new Error('Substrate requires identity, format, capabilities, and requested backends')
    }
  }

  get status(): SubstrateStatus {
    return this.currentStatus
  }

  async load(onProgress?: (progress: Progress) => void, signal?: AbortSignal): Promise<void> {
    if (this.currentStatus !== 'idle') {
      throw new Error(`Cannot load substrate in ${this.currentStatus} state`)
    }
    // Pre-aborted calls must not start a download or allocate any runtime resources.
    if (signal?.aborted) throw loadAborted()
    this.currentStatus = 'loading'
    const promise = (async () => {
      try {
        const instance = await this.substrate.create((progress) => {
          if (this.currentStatus === 'loading' && !signal?.aborted) onProgress?.(progress)
        }, signal)
        if (this.currentStatus !== 'loading' || signal?.aborted) {
          try {
            await instance.dispose()
          } catch (error) {
            this.lateCleanupFailure = { error }
            throw error
          }
          if (signal?.aborted) throw loadAborted()
          throw new Error('Substrate disposed during load')
        }
        this.instance = instance
        this.currentStatus = 'ready'
      } catch (error) {
        if (this.currentStatus === 'loading') {
          this.currentStatus = this.lateCleanupFailure ? 'failed' : 'idle'
        }
        throw error
      }
    })()
    this.loading = promise
    try {
      await promise
    } finally {
      if (this.loading === promise) this.loading = null
    }
  }

  async run(input: I): Promise<O> {
    if (this.currentStatus !== 'ready' || !this.instance) {
      throw new Error(`Cannot infer in ${this.currentStatus} state`)
    }
    this.currentStatus = 'running'
    const promise = Promise.resolve().then(() => this.instance!.infer(input))
    this.running = promise
    try {
      return await promise
    } finally {
      if (this.running === promise) this.running = null
      if (this.currentStatus === 'running') this.currentStatus = 'ready'
    }
  }

  dispose(): Promise<void> {
    if (this.disposing) return this.disposing
    if (this.currentStatus === 'disposed') return Promise.resolve()
    this.currentStatus = 'disposing'
    this.disposing = (async () => {
      try {
        // A late load must release its instance; a running inference must
        // finish before its underlying runtime buffers can be freed.
        await this.loading?.catch(() => undefined)
        if (this.lateCleanupFailure) throw this.lateCleanupFailure.error
        await this.running?.catch(() => undefined)
        const instance = this.instance
        this.instance = null
        if (instance) await instance.dispose()
      } finally {
        this.currentStatus = 'disposed'
      }
    })()
    return this.disposing
  }
}
