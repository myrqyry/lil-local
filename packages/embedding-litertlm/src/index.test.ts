import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { EmbeddingEngine, Backend, getOrLoadGlobalLiteRtLm } from '@litert-lm/core'
import { createEmbeddingGemma2Substrate, formatEmbeddingGemma2Text, EMBEDDING_GEMMA2_MODEL_URL, EMBEDDING_GEMMA2_REVISION } from './index'

vi.mock('@litert-lm/core', () => ({
  Backend: { GPU: 2 },
  EmbeddingEngine: { create: vi.fn() },
  getOrLoadGlobalLiteRtLm: vi.fn(async () => undefined),
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => { resolve = res })
  return { promise, resolve }
}

describe('LiteRT-LM EmbeddingGemma 2 browser adapter', () => {
  beforeEach(() => {
    vi.mocked(EmbeddingEngine.create).mockReset()
    vi.mocked(getOrLoadGlobalLiteRtLm).mockReset()
    vi.mocked(getOrLoadGlobalLiteRtLm).mockResolvedValue(undefined as never)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('chooses a matching 0.18.1 WASM runtime before creating the embedding engine', async () => {
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({
      model: new Blob(['model']),
      wasmBase: '/offline/litert-lm/',
    }))
    vi.mocked(EmbeddingEngine.create).mockResolvedValueOnce({
      delete: vi.fn(async () => undefined),
    } as never)
    await runner.load()
    expect(getOrLoadGlobalLiteRtLm).toHaveBeenCalledWith('/offline/litert-lm/')
    expect(vi.mocked(getOrLoadGlobalLiteRtLm).mock.invocationCallOrder[0])
      .toBeLessThan(vi.mocked(EmbeddingEngine.create).mock.invocationCallOrder[0])
    await runner.dispose()
  })

  it('never fetches a model until explicitly loaded and returns real validated 256d output', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ({
      ok: true,
      body: new ReadableStream<Uint8Array>(),
    }))
    vi.stubGlobal('fetch', fetchMock)
    const computeEmbedding = vi.fn(async () => ({
      embedding: Array.from({ length: 256 }, (_, i) => i / 256),
      inputLength: 12,
      numChunks: 1,
    }))
    const release = vi.fn(async () => undefined)
    vi.mocked(EmbeddingEngine.create).mockResolvedValueOnce({ computeEmbedding, delete: release } as never)
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({
      model: 'https://models.example/embeddinggemma-2.litertlm',
    }))
    expect(runner.status).toBe('idle')
    expect(fetchMock).not.toHaveBeenCalled()
    expect(EmbeddingEngine.create).not.toHaveBeenCalled()

    await runner.load()
    expect(runner.status).toBe('ready')
    expect(fetchMock).toHaveBeenCalledOnce()
    expect(fetchMock.mock.calls[0][0]).toBe('https://models.example/embeddinggemma-2.litertlm')
    expect(vi.mocked(EmbeddingEngine.create)).toHaveBeenCalledWith(expect.objectContaining({
      backend: Backend.GPU,
      model: expect.any(ReadableStream),
    }))
    const result = await runner.run({ text: 'task: search query | text: forest', dimensions: 256 })
    expect(result.kind).toBe('embedding')
    expect(result.dimensions).toBe(256)
    expect(result.values).toBeInstanceOf(Float32Array)
    expect(result.values).toHaveLength(256)
    expect(computeEmbedding).toHaveBeenCalledExactlyOnceWith(
      'task: search query | text: forest',
      { normalize: true, outputSize: 256 },
    )
    await runner.dispose()
    expect(release).toHaveBeenCalledOnce()
  })

  it('aborts the model fetch and never invokes the native engine when cancelled', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn((_url: string, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), {name:'AbortError'})), {once:true})
    }))
    vi.stubGlobal('fetch', fetchMock)
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({
      model: 'https://models.example/embeddinggemma-2.litertlm',
    }))
    const loading = runner.load(undefined, controller.signal)
    controller.abort()
    await expect(loading).rejects.toMatchObject({name:'AbortError'})
    expect(EmbeddingEngine.create).not.toHaveBeenCalled()
    expect(runner.status).toBe('idle')
  })

  it('disposes an engine that finishes after cancellation rather than publishing it', async () => {
    const deferredEngine = deferred<Awaited<ReturnType<typeof EmbeddingEngine.create>>>()
    const release = vi.fn(async () => undefined)
    vi.mocked(EmbeddingEngine.create).mockReturnValueOnce(deferredEngine.promise)
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({model:new Blob(['fixture'])}))
    const controller = new AbortController()
    const loading = runner.load(undefined,controller.signal)
    await vi.waitFor(() => expect(EmbeddingEngine.create).toHaveBeenCalledOnce())
    controller.abort()
    deferredEngine.resolve({delete:release} as never)
    await expect(loading).rejects.toMatchObject({name:'AbortError'})
    expect(release).toHaveBeenCalledOnce()
    expect(runner.status).toBe('idle')
  })

  it('rejects invalid input and malformed native vectors rather than fabricating results', async () => {
    const computeEmbedding = vi.fn(async () => ({
      embedding: [1, 2, Number.NaN],
      inputLength: 2,
      numChunks: 1,
    }))
    vi.mocked(EmbeddingEngine.create).mockResolvedValueOnce({
      computeEmbedding, delete: async () => undefined,
    } as never)
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({model:new Blob()}))
    await runner.load()
    await expect(runner.run({text:'   ',dimensions:128})).rejects.toThrow('nonempty')
    expect(computeEmbedding).not.toHaveBeenCalled()
    await expect(runner.run({text:'valid',dimensions:128})).rejects.toThrow('dimension')
    await runner.dispose()
  })

  it('pins the model URL to an immutable Hub revision without fetching it', () => {
    expect(EMBEDDING_GEMMA2_REVISION).toMatch(/^[0-9a-f]{40}$/)
    expect(EMBEDDING_GEMMA2_MODEL_URL).toContain(
      `/resolve/${EMBEDDING_GEMMA2_REVISION}/embeddinggemma-2-740m.litertlm`,
    )
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({
      model: EMBEDDING_GEMMA2_MODEL_URL,
    }))
    expect(runner.status).toBe('idle')
    expect(EmbeddingEngine.create).not.toHaveBeenCalled()
  })

  it('rejects non-finite values even when the native vector length matches', async () => {
    vi.mocked(EmbeddingEngine.create).mockResolvedValueOnce({
      computeEmbedding: async () => ({
        embedding: [...Array.from({length:127},() => 0), Number.POSITIVE_INFINITY],
        inputLength: 3, numChunks: 1,
      }),
      delete: async () => undefined,
    } as never)
    const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({model:new Blob()}))
    await runner.load()
    await expect(runner.run({text:'test',dimensions:128})).rejects.toThrow('non-finite')
    await runner.dispose()
  })

  it('uses the documented search prefixes explicitly and rejects unsupported tasks', () => {
    expect(formatEmbeddingGemma2Text('  orange cat  ', 'search-query'))
      .toBe('task: search query | text: orange cat')
    expect(formatEmbeddingGemma2Text('orange cat', 'document'))
      .toBe('task: search result | text: orange cat')
    expect(formatEmbeddingGemma2Text('orange cat', 'sentence-similarity'))
      .toBe('task: sentence similarity | text: orange cat')
    expect(() => formatEmbeddingGemma2Text(' ', 'search-query')).toThrow('nonempty')
    expect(() => formatEmbeddingGemma2Text('text', 'unknown' as never)).toThrow('Unsupported')
  })
})
