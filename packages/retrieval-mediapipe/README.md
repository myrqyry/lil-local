# @lil-local/retrieval-mediapipe

A reusable **opt-in** local MediaPipe retrieval substrate. It implements
**Universal Embedder and Semantic Retriever over one loaded EmbeddingGemma 2
model**, following the official web sample's shared-model architecture.
It does not duplicate the existing ColBERT or LiteRT-LM `@lil-local/retrieval`
and `@lil-local/embedding-litertlm` implementations.

Both tasks are exposed through one typed `BrowserSubstrate` so consumers
pay for one model session and one in-memory vector index. Model-loading
is explicit. Importing the package or opening the playground's model list
does not download weights, allocate a GPU, or open the camera/microphone.

## Use it downstream

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  createUniversalRetrievalSubstrate,
  EMBEDDING_GEMMA2_TEXT_270M,
} from '@lil-local/retrieval-mediapipe'

const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
  model: myDownloadedModelFile, // Blob or Uint8Array, or pinned .litertlm URL
  // model: EMBEDDING_GEMMA2_TEXT_270M.url,
  // wasmBase: '/assets/mediapipe-retrieval/wasm', // version-matched 1.2.2
  l2Normalize: true,
}))
await runner.load()

const embedding = await runner.run({kind:'embed-text',text:'small cat in a box'})
// embedding.kind === 'embedding'; Float32Array head vectors with dimensions

await runner.run({kind:'insert-text',id:'cat',text:'A cat sitting in a cardboard box.'})
const matches = await runner.run({kind:'search',query:'cat in a box',limit:5})
// matches.kind === 'search-results'; exact SDK scores and record ids

await runner.dispose() // closes the retriever and shared native embedder
```

The same runner supports `embed-image`, `embed-audio`, `insert-image`,
`insert-audio`, `delete`, `clear` and `list-ids`. Inputs are caller-owned
PNG/JPEG image bytes or decoded normalized mono PCM. **You must select a model
with the correct modalities**: text-only 270M does not magically become a
vision/audio model; 440M Text-Vision and 740M Omnimodal provide those
architectures according to Google's own model cards.

Documents are chunked via MediaPipe's `DefaultTextChunker`. The retriever
uses the *same Universal Embedder provider*, a native
`SemanticRetrieverComponents`, and `MemoryVectorStore`. Index data
exists only in memory for that runner and is destroyed on `dispose()`.
No implicit sync to servers, IndexedDB, or app-specific storage.

The package requires WebGPU for the demonstrated model; the metadata
reports *requested*, not independently attested physical GPU placement.
No silent fallback engine or hidden hosted inference.

## Pinned model for real qualification

The canonical text-only EmbeddingGemma 2 bundle was downloaded outside Git,
at the immutable Hugging Face revision
`9be6e8b90982095dc05c2bd162e4b954ee4dbac7`. Exact bytes:
`164626432`; actual SHA-256:
`2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb`.
These match the publisher's linked-size and linked-etag headers. The
`EMBEDDING_GEMMA2_TEXT_270M` export is a **catalog pin**, not a default
download and not automatic browser-side SHA verification. Consumers supplying
arbitrary models are responsible for provenance.

WASM:
`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-retrieval@1.2.2/wasm`,
aligned with the package's exact `@mediapipe/tasks-retrieval@1.2.2`.
A matching same-version local `wasmBase` is supported.

## Verified and unverified behavior

- Package tests cover opt-in loading, indexed embedding/result shapes, one
  native engine shared with the retriever, correct provider linkage,
  real cleanup order, pre-abort, lazy URL transport, malformed vector/input
  rejection, and cleanup after failed retriever construction.
- The independent pack/consume fixture constructs this substrate using
  only the public `@lil-local/retrieval-mediapipe` entrypoint.
- **Actual Chromium inference** with the pinned 165 MB text bundle:
  WebGPU was available; actual MediaPipe WASM model initialized; returned
  a **768-dimensional** text embedding, indexed a nontrivial text document,
  searched a matching text query and returned `doc-1` at score `0.839`
  in that single run. The retriever was unloaded successfully and
  no browser runtime errors were recorded.
- This text-model smoke does **not** prove real image/audio embeddings,
  multimodal cross-space alignment, latency under sustained load, or
  particular GPU vendor execution. Those need real 440M/740M checkpoints
  and targeted validation before promoting to qualified.
- Current WebGPU implementation may block the main thread. A consumer
  needing sustained frame/audio throughput should use an actual worker
  and explicitly qualify its worker lifecycle. This package is not
  a Web Worker abstraction.

## Reproduce

```bash
pnpm --filter @lil-local/retrieval-mediapipe test
pnpm --filter @lil-local/retrieval-mediapipe typecheck
pnpm test:compatibility
pnpm verify
# Extra: requires the official 164626432-byte model outside Git
pnpm test:retrieval:browser
```

The optional browser smoke reads the model from
`~/.cache/lil-local/mediapipe-retrieval/embeddinggemma-2-text-270m.litertlm`,
or `LIL_LOCAL_RETRIEVAL_MODEL=/absolute/path/model.litertlm`.
It checks the complete local checkpoint SHA-256 before starting Chromium.
The test opens a disposable browser profile, selects a model File through
the playground, explicitly loads the shared runtime, actually embeds,
indexes, retrieves, unloads and deletes the profile. No model download
is implicit in the smoke; weights never enter Git.

The task mapping and detailed outcome are at
`docs/verification/2026-11-04-mediapipe-samples-retrieval.md`.
