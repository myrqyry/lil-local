import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner, type BrowserSubstrate } from './substrate'

const descriptor = {
  id: 'test-onnx',
  format: 'onnx',
  capabilities: ['reasoning'] as const,
  requestedBackends: ['webgpu'] as const,
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

describe('BrowserSubstrateRunner', () => {
  it('does nothing until explicit load and calls real provider inference', async () => {
    const infer = vi.fn(async (input: string) => input.toUpperCase())
    const dispose = vi.fn(async () => undefined)
    const create = vi.fn(async () => ({ infer, dispose }))
    const runner = new BrowserSubstrateRunner({ ...descriptor, create })
    expect(runner.status).toBe('idle')
    expect(create).not.toHaveBeenCalled()
    await expect(runner.run('hello')).rejects.toThrow('idle')
    await runner.load()
    expect(runner.status).toBe('ready')
    expect(await runner.run('hello')).toBe('HELLO')
    expect(infer).toHaveBeenCalledWith('hello')
    await runner.dispose()
    expect(dispose).toHaveBeenCalledTimes(1)
    expect(runner.status).toBe('disposed')
    await expect(runner.run('again')).rejects.toThrow('disposed')
    await runner.dispose()
    expect(dispose).toHaveBeenCalledTimes(1)
  })

  it('forwards only actual load progress and does not implicitly retry or fallback', async () => {
    const onProgress = vi.fn()
    const runner = new BrowserSubstrateRunner({
      ...descriptor,
      async create(progress?: (phase: string) => void) {
        progress?.('graph')
        throw new Error('GPU compilation failed')
      },
    })
    await expect(runner.load(onProgress)).rejects.toThrow('GPU compilation failed')
    expect(onProgress).toHaveBeenCalledExactlyOnceWith('graph')
    expect(runner.status).toBe('idle')
  })

  it('rejects concurrent inference, and holds disposal until inference settles', async () => {
    const pending = deferred<string>()
    const dispose = vi.fn(async () => undefined)
    const runner = new BrowserSubstrateRunner({
      ...descriptor,
      async create() { return { infer: () => pending.promise, dispose } },
    })
    await runner.load()
    const inference = runner.run('first')
    expect(runner.status).toBe('running')
    await expect(runner.run('second')).rejects.toThrow('running')
    const closing = runner.dispose()
    expect(dispose).not.toHaveBeenCalled()
    pending.resolve('ok')
    expect(await inference).toBe('ok')
    await closing
    expect(dispose).toHaveBeenCalledTimes(1)
    expect(runner.status).toBe('disposed')
  })

  it('refuses a pre-aborted load without calling the model loader', async () => {
    const create = vi.fn(async () => ({
      infer: async (x: string) => x, dispose: async () => undefined,
    }))
    const runner = new BrowserSubstrateRunner({ ...descriptor, create })
    const controller = new AbortController()
    controller.abort()
    await expect(runner.load(undefined, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(create).not.toHaveBeenCalled()
    expect(runner.status).toBe('idle')
  })

  it('passes a cancellation signal to the real loader and refuses late readiness after abort', async () => {
    const controller = new AbortController()
    const pending = deferred<{ infer: (input: string) => Promise<string>; dispose: () => Promise<void> }>()
    const dispose = vi.fn(async () => undefined)
    const receivedSignal = vi.fn()
    const runner = new BrowserSubstrateRunner({
      ...descriptor,
      create: async (_progress, signal) => {
        receivedSignal(signal)
        return pending.promise
      },
    })
    const progress = vi.fn()
    const loading = runner.load(progress, controller.signal)
    expect(runner.status).toBe('loading')
    controller.abort()
    pending.resolve({ infer: async (text) => text, dispose })
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(receivedSignal).toHaveBeenCalledExactlyOnceWith(controller.signal)
    expect(dispose).toHaveBeenCalledOnce()
    expect(runner.status).toBe('idle')
    await runner.dispose()
    expect(dispose).toHaveBeenCalledOnce()
  })

  it('does not forward late progress after abort and does not mask a failed cancellation cleanup', async () => {
    const controller = new AbortController()
    const pending = deferred<{ infer: (input: string) => Promise<string>; dispose: () => Promise<void> }>()
    let publishProgress: (value: string) => void = () => undefined
    const onProgress = vi.fn()
    const runner = new BrowserSubstrateRunner({
      ...descriptor,
      create: async (report?: (value: string) => void) => {
        publishProgress = report ?? publishProgress
        return pending.promise
      },
    })
    const loading = runner.load(onProgress, controller.signal)
    publishProgress('fetching')
    controller.abort()
    publishProgress('ready')
    pending.resolve({
      infer: async (text) => text,
      dispose: async () => { throw new Error('resource cleanup failed') },
    })
    await expect(loading).rejects.toThrow('resource cleanup failed')
    expect(onProgress).toHaveBeenCalledExactlyOnceWith('fetching')
    expect(runner.status).toBe('failed')
    await expect(runner.load()).rejects.toThrow('failed')
    await expect(runner.dispose()).rejects.toThrow('resource cleanup failed')
    expect(runner.status).toBe('disposed')
  })

  it('releases late-loaded instances rather than publishing a ready state', async () => {
    const pending = deferred<{ infer: (input: string) => Promise<string>; dispose: () => Promise<void> }>()
    const dispose = vi.fn(async () => undefined)
    const runner = new BrowserSubstrateRunner({ ...descriptor, create: () => pending.promise })
    const loading = runner.load()
    const closing = runner.dispose()
    pending.resolve({ infer: async x => x, dispose })
    await expect(loading).rejects.toThrow('disposed during load')
    await closing
    expect(dispose).toHaveBeenCalledTimes(1)
    expect(runner.status).toBe('disposed')
  })

  it('reports failed late-resource cleanup instead of masking it as success', async () => {
    const pending = deferred<{ infer: (input: string) => Promise<string>; dispose: () => Promise<void> }>()
    const runner = new BrowserSubstrateRunner({ ...descriptor, create: () => pending.promise })
    const loading = runner.load()
    const closing = runner.dispose()
    pending.resolve({
      infer: async text => text,
      dispose: async () => { throw new Error('GPU release failed') },
    })
    await expect(loading).rejects.toThrow('GPU release failed')
    await expect(closing).rejects.toThrow('GPU release failed')
    expect(runner.status).toBe('disposed')
  })

  it('recovers from inference exceptions without pretending they succeeded', async () => {
    const runner = new BrowserSubstrateRunner({
      ...descriptor,
      async create() {
        return {
          infer: async () => { throw new Error('native runtime lost') },
          dispose: async () => undefined,
        }
      },
    })
    await runner.load()
    await expect(runner.run('fail')).rejects.toThrow('native runtime lost')
    expect(runner.status).toBe('ready')
    await runner.dispose()
  })

  it('requires an identified provider with capabilities and backends', () => {
    const invalid: BrowserSubstrate<string, string> = {
      id: '', format: 'gguf', capabilities: [], requestedBackends: [],
      create: async () => ({ infer: async text => text, dispose: async () => undefined }),
    }
    expect(() => new BrowserSubstrateRunner(invalid)).toThrow('requires identity')
  })
})
