import { describe, expect, it } from 'vitest'
import { BrowserSubstrateRunner, type BrowserSubstrate } from '@lil-local/inference-core'

const fixture = (onDispose: () => void = () => {}) => ({
  id: 'external-packed-consumer',
  format: 'fixture',
  capabilities: ['text-generation'],
  requestedBackends: ['wasm'],
  async create() {
    return { infer: async (input: string) => input.toUpperCase(), dispose: async () => onDispose() }
  },
}) satisfies BrowserSubstrate<string, string>

describe('installed packed inference-core contract', () => {
  it('loads, executes, and disposes through the public entrypoint', async () => {
    let disposals = 0
    const runner = new BrowserSubstrateRunner(fixture(() => disposals++))
    expect(runner.status).toBe('idle')
    await runner.load()
    expect(runner.status).toBe('ready')
    expect(await runner.run('hello')).toBe('HELLO')
    await runner.dispose()
    expect(runner.status).toBe('disposed')
    expect(disposals).toBe(1)
  })

  it('does not construct the model when pre-aborted', async () => {
    let creates = 0
    const descriptor = { ...fixture(), create: async () => {
      creates++
      return { infer: async (input: string) => input, dispose: async () => {} }
    }}
    const runner = new BrowserSubstrateRunner(descriptor)
    const controller = new AbortController()
    controller.abort()
    await expect(runner.load(undefined, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(creates).toBe(0)
    expect(runner.status).toBe('idle')
  })

  it('propagates inference failures without swapping in a CPU result', async () => {
    const runner = new BrowserSubstrateRunner({
      ...fixture(),
      create: async () => ({ infer: async (_input: string): Promise<string> => { throw new Error('GPU failed') }, dispose: async () => {} }),
    })
    await runner.load()
    await expect(runner.run('hello')).rejects.toThrow('GPU failed')
    expect(runner.status).toBe('ready')
    await runner.dispose()
  })
})
