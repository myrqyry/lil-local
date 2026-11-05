import { beforeEach, describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  FilesetResolver, UniversalEmbedder, DefaultTextChunker, MemoryVectorStore,
  SemanticRetriever, SemanticRetrieverComponents,
} from '@mediapipe/tasks-retrieval'
import {
  createUniversalRetrievalSubstrate, validateUniversalResult,
  validateRetrievalRequest, RETRIEVAL_WASM_BASE, EMBEDDING_GEMMA2_TEXT_270M,
} from './index'

vi.mock('@mediapipe/tasks-retrieval', () => ({
  FilesetResolver: { forRetrievalTasks: vi.fn(async () => ({ wasm: true })) },
  UniversalEmbedder: { createFromOptions: vi.fn() },
  DefaultTextChunker: { create: vi.fn(), createFromModule: vi.fn() },
  MemoryVectorStore: vi.fn(),
  SemanticRetriever: { createFromComponents: vi.fn() },
  SemanticRetrieverComponents: vi.fn(),
}))

function mockSdk() {
  const events: string[] = []
  const provider = { embedContent: vi.fn(async () => Float32Array.of(1, 2, 3)) }
  const embedText = vi.fn(async () => ({ embeddings: [{
    headIndex: 0, headName: 'main', floatEmbedding: [0.25, 0.5, 0.75],
  }] }))
  const embedImage = vi.fn(async () => ({ embeddings: [{
    headIndex: 0, headName: 'vision', floatEmbedding: [0.75, 0.5, 0.25],
  }] }))
  const embedAudio = vi.fn(async () => ({ embeddings: [{
    headIndex: 0, headName: 'audio', floatEmbedding: [0.5, 0.3, 0.2],
  }] }))
  const embedder = {
    embedText, embedImage, embedAudio,
    getProvider: vi.fn(() => provider),
    close: vi.fn(() => events.push('embedder')),
  }
  const retriever = {
    insertDocument: vi.fn(async () => {}),
    insertImage: vi.fn(async () => {}),
    insertAudio: vi.fn(async () => {}),
    retrieve: vi.fn(async () => [{id:'cat',score:0.91,metadata:{source:'local'},
      content:[{text:'a cat in a box'}]}]),
    delete: vi.fn(async () => {}),
    deleteAll: vi.fn(async () => {}),
    getAllRecordIds: vi.fn(async () => ['cat']),
    close: vi.fn(() => events.push('retriever')),
  }
  const store = {close: vi.fn(() => events.push('store'))}
  const chunker = {}
  const components = {
    addProvider: vi.fn().mockReturnThis(),
    setVectorStore: vi.fn().mockReturnThis(),
    setTextChunker: vi.fn().mockReturnThis(),
  }
  vi.mocked(UniversalEmbedder.createFromOptions).mockResolvedValue(embedder as never)
  vi.mocked(DefaultTextChunker.create).mockResolvedValue(chunker as never)
  vi.mocked(MemoryVectorStore).mockImplementation(function () { return store as never })
  vi.mocked(SemanticRetrieverComponents).mockImplementation(function () { return components as never })
  vi.mocked(SemanticRetriever.createFromComponents).mockResolvedValue(retriever as never)
  return { events, embedder, retriever, store, components, provider, embedText, embedImage, embedAudio }
}

describe('MediaPipe Universal Embedder + Semantic Retriever as one shared substrate', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(FilesetResolver.forRetrievalTasks).mockResolvedValue({ wasm: true } as never)
  })

  it('is cold until load and pins the same SDK/WASM version', () => {
    const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
      model: new Blob(['local bundle']),
    }))
    expect(runner.status).toBe('idle')
    expect(runner.substrate.capabilities).toContain('text-embedding')
    expect(runner.substrate.capabilities).toContain('image-embedding')
    expect(runner.substrate.capabilities).toContain('audio-embedding')
    expect(RETRIEVAL_WASM_BASE).toContain('@mediapipe/tasks-retrieval@1.2.2/wasm')
    expect(EMBEDDING_GEMMA2_TEXT_270M.url).toContain('/resolve/'+EMBEDDING_GEMMA2_TEXT_270M.revision+'/')
    expect(EMBEDDING_GEMMA2_TEXT_270M.bytes).toBe(164626432)
    expect(EMBEDDING_GEMMA2_TEXT_270M.sha256).toBe('2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb')
    expect(UniversalEmbedder.createFromOptions).not.toHaveBeenCalled()
  })

  it('loads one embedding engine, uses its provider for retrieval, and releases both', async () => {
    const sdk = mockSdk()
    const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
      model: new Uint8Array([3, 2, 1]),
      wasmBase: '/local/mediapipe-retrieval/wasm',
    }))
    await runner.load()
    expect(FilesetResolver.forRetrievalTasks).toHaveBeenCalledWith('/local/mediapipe-retrieval/wasm')
    expect(UniversalEmbedder.createFromOptions).toHaveBeenCalledOnce()
    expect(sdk.components.addProvider).toHaveBeenCalledExactlyOnceWith(sdk.provider)
    const embedding = await runner.run({ kind: 'embed-text', text: '  a cat in a box ' })
    expect(embedding).toMatchObject({
      kind: 'embedding', modality: 'text', embeddings: [{
        headName:'main', values: Float32Array.from([.25,.5,.75]), dimensions:3,
      }],
    })
    expect(sdk.embedText).toHaveBeenCalledWith('a cat in a box')
    expect(await runner.run({kind:'embed-image',imageBytes:Uint8Array.of(2,3)}))
      .toMatchObject({kind:'embedding',modality:'image'})
    expect(sdk.embedImage).toHaveBeenCalledWith(Uint8Array.of(2,3))
    expect(await runner.run({kind:'embed-audio',audioSamples:Float32Array.of(.1,.2)}))
      .toMatchObject({kind:'embedding',modality:'audio'})
    expect(sdk.embedAudio).toHaveBeenCalledWith(Float32Array.of(.1,.2))

    await runner.run({kind:'insert-text',id:'cat',text:'a cat in a box',metadata:{source:'local'}})
    expect(sdk.retriever.insertDocument).toHaveBeenCalledWith('cat','a cat in a box',{source:'local'})
    const search = await runner.run({kind:'search',query:'cat',limit:4})
    expect(sdk.retriever.retrieve).toHaveBeenCalledWith('cat',{limit:4})
    expect(search).toMatchObject({kind:'search-results',matches:[{id:'cat',score:.91}]})
    await runner.dispose()
    expect(sdk.events.indexOf('retriever')).toBeGreaterThanOrEqual(0)
    expect(sdk.events.indexOf('embedder')).toBeGreaterThan(sdk.events.indexOf('retriever'))
    expect(UniversalEmbedder.createFromOptions).toHaveBeenCalledTimes(1)
  })

  it('downloads only on explicit load, forwards abort, and checks status', async () => {
    const sdk=mockSdk()
    const fetcher = vi.fn(async (_url: string, _opts: RequestInit) =>
      ({ok:true,body:new ReadableStream<Uint8Array>(),
        status:200}) as Response)
    vi.stubGlobal('fetch',fetcher)
    try {
      const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
        model:'https://models.example/embeddinggemma.litertlm',
      }))
      expect(fetcher).not.toHaveBeenCalled()
      await runner.load()
      expect(fetcher).toHaveBeenCalledOnce()
      expect(fetcher.mock.calls[0][1]).toHaveProperty('signal')
      expect(sdk.embedder.getProvider).toHaveBeenCalledOnce()
      await runner.dispose()
    } finally {vi.unstubAllGlobals()}
  })

  it('rejects pre-aborted load without fetching or creating a runtime', async () => {
    const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
      model:'https://models.example/embeddinggemma.litertlm',
    }))
    const abort=new AbortController(); abort.abort()
    await expect(runner.load(undefined,abort.signal))
      .rejects.toMatchObject({name:'AbortError'})
    expect(UniversalEmbedder.createFromOptions).not.toHaveBeenCalled()
  })

  it('rejects invalid options and malformed vectors instead of fabricating embeddings', () => {
    expect(()=>createUniversalRetrievalSubstrate({model:''})).toThrow(/model/i)
    expect(()=>createUniversalRetrievalSubstrate({model:new Uint8Array()})).toThrow(/empty/i)
    expect(()=>createUniversalRetrievalSubstrate({model:new Blob()})).toThrow(/empty/i)
    expect(()=>createUniversalRetrievalSubstrate({
      model:new Blob(['x']),chunkSize:20,chunkOverlap:20,
    })).toThrow(/overlap/i)
    expect(()=>validateUniversalResult({embeddings:[]})).toThrow(/embedding/i)
    expect(()=>validateUniversalResult({embeddings:[{
      headIndex:0,headName:'',floatEmbedding:[1,Number.NaN],
    }]})).toThrow(/finite/i)
    expect(()=>validateUniversalResult({embeddings:[{
      headIndex:0,headName:'',quantizedEmbedding:Uint8Array.of(9),
    }]})).toThrow(/quantized/i)
  })

  it('rejects invalid and unbounded requests before invoking the native engine', async () => {
    expect(()=>validateRetrievalRequest({kind:'embed-text',text:'  '})).toThrow(/text/i)
    expect(()=>validateRetrievalRequest({kind:'embed-image',imageBytes:new Uint8Array()})).toThrow(/image/i)
    expect(()=>validateRetrievalRequest({kind:'embed-audio',audioSamples:Float32Array.of(NaN)})).toThrow(/finite/i)
    expect(()=>validateRetrievalRequest({kind:'insert-text',id:'',text:'cat'})).toThrow(/id/i)
    expect(()=>validateRetrievalRequest({kind:'search',query:'cat',limit:0})).toThrow(/limit/i)
    expect(()=>validateRetrievalRequest({kind:'search',query:'cat',limit:5000})).toThrow(/limit/i)
    expect(()=>validateRetrievalRequest({kind:'search',query:'  '})).toThrow(/query/i)
  })

  it('closes newly loaded embedding engine when retriever construction fails', async () => {
    const sdk=mockSdk()
    vi.mocked(SemanticRetriever.createFromComponents).mockRejectedValueOnce(new Error('native index failed'))
    const runner=new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({model:new Blob(['data'])}))
    await expect(runner.load()).rejects.toThrow('native index failed')
    expect(sdk.embedder.close).toHaveBeenCalledOnce()
    expect(runner.status).toBe('idle')
  })
})
