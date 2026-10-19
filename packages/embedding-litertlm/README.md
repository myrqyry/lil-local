# EmbeddingGemma 2 via LiteRT-LM Web (experimental)

A reusable **text-only** browser adapter using Google's actual
`@litert-lm/core@0.18.1` `EmbeddingEngine`, exposed through Lil Local's
`BrowserSubstrateRunner`. Nothing is downloaded on import or construction.

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  createEmbeddingGemma2Substrate,
  formatEmbeddingGemma2Text,
  EMBEDDING_GEMMA2_MODEL_URL,
} from '@lil-local/embedding-litertlm'

const runner = new BrowserSubstrateRunner(createEmbeddingGemma2Substrate({
  model: EMBEDDING_GEMMA2_MODEL_URL, // Immutable Hub revision; not downloaded yet
}))
const controller = new AbortController()
await runner.load(undefined, controller.signal) // EXPLICIT large model download
try {
  const result = await runner.run({
    text: formatEmbeddingGemma2Text('red bicycle', 'search-query'),
    dimensions: 256,
  })
  console.log(result.values) // Float32Array, 256 dimensions
} finally {
  await runner.dispose()
}
```

**WASM version guard:** `@litert-lm/core@0.18.1` still embeds a default
URL for `@litert-lm/core@0.12.1/wasm`. This adapter explicitly loads
the matching `0.18.1` JS/WASM bundle before model creation; it fails
rather than silently inheriting a runtime initialized from a conflicting path.
By default, the matching runtime comes from jsDelivr (network required).
For local-first/offline deployments, copy **all four JS/WASM variants** from
the installed `@litert-lm/core@0.18.1/wasm/` package into a served directory
and pass `wasmBase: '/litert-lm-wasm/'`. Do not mix SDK and WASM versions;
nor reinitialize a shared JS realm from a different base.

The adapter uses WebGPU explicitly and enables only the text encoder. The
model source is caller-supplied; the exported Hub URL is a convenient
revision-pinned source but not a verified file SHA-256. For HTTP(S) inputs the
adapter streams `fetch(url, {signal})` into `EmbeddingEngine.create()`,
avoiding a full-checkpoint ArrayBuffer copy. A caller-owned Blob or stream
is passed through unchanged. Aborting a network fetch is supported;
interrupting compilation already inside LiteRT-LM is *not* guaranteed.
The common runner releases an engine returned after cancellation.

Only the model's documented Matryoshka sizes (128, 256, 512, 768) are allowed;
native computation is requested with `normalize: true`, and outputs are
validated for shape and finite values. Query and document task prefixes are
explicit, matching the **LiteRT-LM embedding models guide**. Do not mix
embedding models, versions, output dimensions, or task prefix conventions
without re-indexing and validating retrieval quality.

**Qualification boundary:** Tests use a controlled engine implementation
to validate invocation/cancellation/output contracts. Packed downstream
typechecking and Vite bundling verify portability. No EmbeddingGemma 2
weights were downloaded or executed in Chromium, no numerical-parity or
cross-modal retrieval test has passed, and no model is marked browser-qualified.

Sources:
- https://developers.google.com/edge/litert-lm/embedding_models
- https://developers.google.com/edge/litert-lm/js
- https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm
