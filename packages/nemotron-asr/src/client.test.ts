import { describe, expect, it, vi } from 'vitest'
import { NemotronClient, chooseNemotronBackend, MODEL_REVISION, validateAudioSamples } from './index'

class FakeWorker {
  onmessage: ((message: MessageEvent) => void) | null = null
  messages: Array<{ message: any; transfers: Transferable[] }> = []
  terminated = false
  postMessage(message: any, transfers: Transferable[] = []) {
    this.messages.push({ message, transfers })
  }
  reply(id: number, type: string, value?: unknown) {
    this.onmessage?.({ data: { id, type, value } } as MessageEvent)
  }
  terminate() { this.terminated = true }
}

describe('public Nemotron ONNX ASR consumer contract', () => {
  it('exposes the existing immutable model and explicit local backend policy', () => {
    expect(MODEL_REVISION).toBe('8364d9e2dd9da23789b480bdbba9e423717e42ee')
    expect(chooseNemotronBackend(false, 'auto')).toBe('wasm')
    expect(() => chooseNemotronBackend(false, 'webgpu')).toThrow(/WebGPU/)
    expect(() => validateAudioSamples(Float32Array.of(Number.NaN))).toThrow(/finite/)
  })

  it('creates a cold worker, loads on explicit request, and disposes it', async () => {
    const worker = new FakeWorker()
    const fetch = vi.spyOn(globalThis, 'fetch')
    const client = new NemotronClient({
      workerFactory: () => worker as unknown as Worker,
      assets: { jointModelUrl: '/models/nemotron-joint.onnx' },
    })
    expect(worker.messages).toHaveLength(0)
    expect(fetch).not.toHaveBeenCalled()
    const statuses: string[] = []
    const pending = client.load(message => statuses.push(message), 'wasm')
    const id = worker.messages[0].message.id
    expect(worker.messages[0].message).toMatchObject({ type: 'load', backend: 'wasm', assets: { jointModelUrl: '/models/nemotron-joint.onnx' } })
    worker.reply(id, 'status', undefined)
    worker.onmessage?.({ data: { id, type: 'status', message: 'loading encoder' } } as MessageEvent)
    worker.reply(id, 'result', undefined)
    await pending
    expect(statuses).toContain('loading encoder')
    const disposal = client.dispose()
    const disposeId = worker.messages[worker.messages.length - 1].message.id
    worker.reply(disposeId, 'result')
    await disposal
    expect(worker.terminated).toBe(true)
    fetch.mockRestore()
  })

  it('copies caller PCM so worker transfers never detach consumer-owned audio', async () => {
    const worker = new FakeWorker()
    const client = new NemotronClient({ workerFactory: () => worker as unknown as Worker })
    const pcm = Float32Array.of(0.25, -0.25)
    const pending = client.pushAudio(pcm)
    const request = worker.messages[0]
    expect(request.message.type).toBe('pushAudio')
    expect(request.message.samples).not.toBe(pcm.buffer)
    expect(Array.from(new Float32Array(request.message.samples))).toEqual(Array.from(pcm))
    expect(request.transfers).toContain(request.message.samples)
    worker.reply(request.message.id, 'result', { chunks: 0, text: '', elapsedMs: 0 })
    await pending
    expect(Array.from(pcm)).toEqual([0.25, -0.25])
    const disposal = client.dispose()
    worker.reply(worker.messages[worker.messages.length - 1].message.id, 'result')
    await disposal
  })
})
