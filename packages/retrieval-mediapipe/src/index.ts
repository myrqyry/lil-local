import type { BrowserSubstrate } from '@lil-local/inference-core'
import type {
  Embedding, UniversalEmbedderResult, RetrievalResult, SemanticRetriever,
} from '@mediapipe/tasks-retrieval'

/**
 * Exact SDK/WASM version pairing. Self-host these same 1.2.2 assets with
 * wasmBase to avoid the CDN at runtime.
 */
export const RETRIEVAL_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-retrieval@1.2.2/wasm'

/** Real downloaded official text-only bundle; provenance, not an auto-loading default. */
export const EMBEDDING_GEMMA2_TEXT_270M = {
  revision: '9be6e8b90982095dc05c2bd162e4b954ee4dbac7',
  url: 'https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/9be6e8b90982095dc05c2bd162e4b954ee4dbac7/embeddinggemma-2-text-270m.litertlm',
  bytes: 164_626_432,
  sha256: '2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb',
  modalities: ['text'] as const,
} as const

/** Caller-owned local model or an explicit URL. Never a mutable default model. */
export type RetrievalModelSource = string | Blob | Uint8Array
export type RetrievalModality = 'text' | 'image' | 'audio'

export interface UniversalRetrievalOptions {
  model: RetrievalModelSource
  wasmBase?: string
  l2Normalize?: boolean
  /** Number of characters in one chunk; only used for indexed documents. */
  chunkSize?: number
  chunkOverlap?: number
  maxInputLength?: number
  visionTokensPerImage?: number
}

export type UniversalRetrievalInput =
  | {kind:'embed-text'; text:string}
  | {kind:'embed-image'; imageBytes:Uint8Array}
  | {kind:'embed-audio'; audioSamples:Float32Array}
  | {kind:'insert-text'; id:string; text:string; metadata?:Record<string,string>}
  | {kind:'insert-image'; id:string; imageBytes:Uint8Array; metadata?:Record<string,string>}
  | {kind:'insert-audio'; id:string; audioSamples:Float32Array; metadata?:Record<string,string>}
  | {kind:'search'; query:string; limit?:number; metadataFilter?:Record<string,string>}
  | {kind:'delete'; ids:string[]}
  | {kind:'clear'}
  | {kind:'list-ids'}

export interface MediaPipeFloatEmbedding {
  headIndex: number
  headName: string
  dimensions: number
  values: Float32Array
}
export type UniversalRetrievalOutput =
  | {kind:'embedding'; modality:RetrievalModality; embeddings:MediaPipeFloatEmbedding[]}
  | {kind:'search-results'; matches:RetrievalResult[]}
  | {kind:'ids'; ids:string[]}
  | {kind:'mutated'; action:'insert'|'delete'|'clear'}

function nonempty(value: string, label: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(label+' must be nonempty')
  return value.trim()
}
function checkBinary(data: Uint8Array, label: string) {
  if (!(data instanceof Uint8Array) || !data.byteLength)
    throw new Error(label+' must have nonempty Uint8Array bytes')
}
function checkPcm(data: Float32Array) {
  if (!(data instanceof Float32Array) || !data.byteLength)
    throw new Error('Audio must contain nonempty Float32 PCM')
  for (const n of data) {
    if (!Number.isFinite(n)) throw new Error('Audio PCM must be finite')
    if (n < -1 || n > 1) throw new Error('Audio PCM must be normalized [-1,1]')
  }
}
function checkMetadata(metadata?: Record<string,string>) {
  if (metadata === undefined) return
  if (!metadata || Array.isArray(metadata) || typeof metadata !== 'object' ||
      Object.entries(metadata).some(([key, val]) => !key || typeof val !== 'string'))
    throw new Error('Metadata must be a string-to-string record')
}

/** Guard pre-inference inputs, including dangerous unbounded search limits. */
export function validateRetrievalRequest(input: UniversalRetrievalInput): void {
  if (!input || typeof input !== 'object') throw new Error('Retrieval input required')
  switch (input.kind) {
    case 'embed-text': nonempty(input.text,'Embedding text'); break
    case 'embed-image': checkBinary(input.imageBytes,'Embedding image'); break
    case 'embed-audio': checkPcm(input.audioSamples); break
    case 'insert-text':
      nonempty(input.id,'Record id'); nonempty(input.text,'Document text')
      checkMetadata(input.metadata); break
    case 'insert-image':
      nonempty(input.id,'Record id'); checkBinary(input.imageBytes,'Record image')
      checkMetadata(input.metadata); break
    case 'insert-audio':
      nonempty(input.id,'Record id'); checkPcm(input.audioSamples)
      checkMetadata(input.metadata); break
    case 'search':
      nonempty(input.query,'Search query')
      if (input.limit !== undefined &&
          (!Number.isInteger(input.limit) || input.limit < 1 || input.limit > 100))
        throw new Error('Search limit must be an integer between 1 and 100')
      checkMetadata(input.metadataFilter); break
    case 'delete':
      if (!Array.isArray(input.ids) || input.ids.length === 0 ||
          input.ids.some(id => typeof id !== 'string' || !id.trim()))
        throw new Error('Delete requires nonempty record ids')
      break
    case 'clear':
    case 'list-ids': break
    default: throw new Error('Unsupported MediaPipe retrieval request')
  }
}

/** Never silently turn a quantized vector into a bogus float embedding. */
export function validateUniversalResult(
  output: Pick<UniversalEmbedderResult, 'embeddings'>,
): MediaPipeFloatEmbedding[] {
  if (!output?.embeddings?.length) throw new Error('MediaPipe returned no embeddings')
  return output.embeddings.map((embedding: Embedding) => {
    if (!embedding.floatEmbedding) {
      if (embedding.quantizedEmbedding) throw new Error('Quantized embeddings are not supported by this float-vector API')
      throw new Error('MediaPipe embedding is missing float values')
    }
    const values = embedding.floatEmbedding
    if (!values.length) throw new Error('MediaPipe embedding is empty')
    if (values.some(n => !Number.isFinite(n))) throw new Error('MediaPipe embedding values must be finite')
    return {
      headIndex: embedding.headIndex,
      headName: embedding.headName,
      dimensions: values.length,
      values: Float32Array.from(values),
    }
  })
}
function abortError(): Error {
  const err = new Error('MediaPipe retrieval load aborted')
  err.name = 'AbortError'
  return err
}
function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError()
}
function validateOptions(options: UniversalRetrievalOptions) {
  if (!options?.model || (typeof options.model === 'string' && !options.model.trim()) ||
      (options.model instanceof Blob && !options.model.size) ||
      (options.model instanceof Uint8Array && !options.model.byteLength))
    throw new Error('MediaPipe retrieval requires a nonempty explicit model')
  const chunkSize = options.chunkSize ?? 512
  const chunkOverlap = options.chunkOverlap ?? 100
  if (!Number.isSafeInteger(chunkSize) || chunkSize < 2)
    throw new Error('Retrieval chunkSize must be an integer >= 2')
  if (!Number.isSafeInteger(chunkOverlap) || chunkOverlap < 0 || chunkOverlap >= chunkSize)
    throw new Error('Retrieval chunkOverlap must be smaller than chunkSize')
  for (const key of ['maxInputLength','visionTokensPerImage'] as const) {
    const val=options[key]
    if (val !== undefined && (!Number.isSafeInteger(val) || val <= 0))
      throw new Error(key+' must be a positive integer')
  }
  if (options.wasmBase !== undefined && !options.wasmBase.trim())
    throw new Error('MediaPipe retrieval wasmBase must be nonempty')
}

async function loadModelSource(model: RetrievalModelSource, signal?: AbortSignal) {
  checkAbort(signal)
  if (model instanceof Uint8Array) return {modelAssetBuffer:model}
  if (model instanceof Blob) return {modelAssetBuffer:model.stream().getReader()}
  // Fetch with AbortSignal, allowing streaming model bodies without an extra 388+MB copy.
  const response = await fetch(model, {signal})
  if (!response.ok) throw new Error('MediaPipe retrieval model fetch failed: HTTP '+response.status)
  if (!response.body) throw new Error('MediaPipe retrieval model has no response body')
  checkAbort(signal)
  return {modelAssetBuffer:response.body.getReader()}
}

/**
 * A single model load backs both universal multimodal embeddings and semantic
 * retrieval. The MemoryVectorStore is in-memory and never persisted implicitly.
 * Consumer owns model choice, cache transport, and any product/application logic.
 */
export function createUniversalRetrievalSubstrate(
  options: UniversalRetrievalOptions,
): BrowserSubstrate<UniversalRetrievalInput, UniversalRetrievalOutput> {
  validateOptions(options)
  return {
    id:'mediapipe:universal-embedding-retrieval',
    format:'litertlm',
    capabilities:['text-embedding','image-embedding','audio-embedding','semantic-retrieval'],
    requestedBackends:['webgpu'],
    async create(_progress, signal) {
      checkAbort(signal)
      const sdk=await import('@mediapipe/tasks-retrieval')
      checkAbort(signal)
      const wasm=await sdk.FilesetResolver.forRetrievalTasks(options.wasmBase??RETRIEVAL_WASM_BASE)
      checkAbort(signal)
      const model=await loadModelSource(options.model, signal)
      checkAbort(signal)

      const embedder=await sdk.UniversalEmbedder.createFromOptions(wasm,{
        baseOptions:model,
        l2Normalize:options.l2Normalize??true,
        maxInputLength:options.maxInputLength,
        visionTokensPerImage:options.visionTokensPerImage,
      })
      let retriever:SemanticRetriever|undefined
      let store:InstanceType<typeof sdk.MemoryVectorStore>|undefined
      try {
        checkAbort(signal)
        // Use the same native Wasm module when the SDK exposes it; otherwise
        // the documented fileset path initializes just the chunker separately.
        const internal=embedder as unknown as {wasmModule?: Parameters<typeof sdk.DefaultTextChunker.createFromModule>[0]}
        const chunker=internal.wasmModule
          ? sdk.DefaultTextChunker.createFromModule(internal.wasmModule, options.chunkSize??512,options.chunkOverlap??100)
          : await sdk.DefaultTextChunker.create(wasm,options.chunkSize??512,options.chunkOverlap??100)
        checkAbort(signal)
        store=new sdk.MemoryVectorStore()
        const components=new sdk.SemanticRetrieverComponents()
          .addProvider(embedder.getProvider())
          .setVectorStore(store)
          .setTextChunker(chunker)
        retriever=await sdk.SemanticRetriever.createFromComponents(components)
        checkAbort(signal)
        // Fully constructed: ownership moves to the loaded runner.
        const activeRetriever=retriever
        return {
          async infer(input) {
            validateRetrievalRequest(input)
            switch (input.kind) {
              case 'embed-text': return {
                kind:'embedding', modality:'text',
                embeddings:validateUniversalResult(await embedder.embedText(input.text.trim())),
              }
              case 'embed-image': return {
                kind:'embedding',modality:'image',
                embeddings:validateUniversalResult(await embedder.embedImage(input.imageBytes)),
              }
              case 'embed-audio': return {
                kind:'embedding',modality:'audio',
                embeddings:validateUniversalResult(await embedder.embedAudio(input.audioSamples)),
              }
              case 'insert-text':
                await activeRetriever.insertDocument(input.id,input.text,input.metadata)
                return {kind:'mutated',action:'insert'}
              case 'insert-image':
                await activeRetriever.insertImage(input.id,input.imageBytes,input.metadata)
                return {kind:'mutated',action:'insert'}
              case 'insert-audio':
                await activeRetriever.insertAudio(input.id,input.audioSamples,input.metadata)
                return {kind:'mutated',action:'insert'}
              case 'search':
                return {kind:'search-results',
                  matches:await activeRetriever.retrieve(input.query,{
                    ...(input.limit===undefined?{}:{limit:input.limit}),
                    ...(input.metadataFilter===undefined?{}:{metadataFilter:input.metadataFilter}),
                  })}
              case 'delete':
                await activeRetriever.delete(input.ids)
                return {kind:'mutated',action:'delete'}
              case 'clear':
                await activeRetriever.deleteAll()
                return {kind:'mutated',action:'clear'}
              case 'list-ids':
                return {kind:'ids',ids:await activeRetriever.getAllRecordIds()}
            }
          },
          async dispose() {
            try { activeRetriever.close() }
            finally { embedder.close() }
          },
        }
      } catch(error) {
        try { retriever?.close() ?? store?.close() }
        finally { embedder.close() }
        throw error
      }
    },
  }
}
