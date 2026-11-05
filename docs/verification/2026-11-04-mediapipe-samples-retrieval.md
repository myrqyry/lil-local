# Official MediaPipe web samples → Lil Local retrieval — 2026-11-04

## Source and motivation

Source: [google-ai-edge/mediapipe-samples-web](https://github.com/google-ai-edge/mediapipe-samples-web),
cloned and examined at immutable upstream commit
`6887f6399745dc549dadd0022c31fc9ab01f26d6` on 2026-11-04.
Key source files:

- `src/components/retrieval-runtime.ts` — one shared `UniversalEmbedder`
  loaded from an explicit File or URL, reused by the similarity UI,
  `SemanticRetriever`, and the Decision Maker demo; in-memory
  `MemoryVectorStore` and `DefaultTextChunker` built around that provider.
- `src/tasks/universal-embedder.ts` — real `embedText`, `embedImage`,
  cosine similarity, caller-selected model and output display.
- `src/tasks/semantic-retriever.ts` — insert image/text corpus, query,
  return scored results.
- `src/components/model-registry.ts` — text 270M, text+vision 440M,
  omnimodal 740M, and other model families.
- `tests/universal-embedder.spec.ts` and
  `tests/semantic-retriever.spec.ts` — upstream Playwright expectations.

**Architectural interpretation:** Lil Local is the shared browser inference
substrate. Add a reusable **one-model embedding + retrieval capability**;
do not export a demo component or duplicate the native engine in each app.

## Where the other upstream samples land

| Upstream sample lane | Lil Local status / decision |
| --- | --- |
| Face Landmarker + Gesture Recognizer | Existing `@lil-local/vision-mediapipe`, with real model and fake-camera Chromium inference evidence. |
| Face/Hand/Holistic/Pose Landmarker, object detector and related Vision tasks | Distinct SDK task families; no claims of support until their separate assets, outputs and browser qualification are added. Existing image-segmentation/depth/image-embedding packages are **different implementations**, not aliases. |
| Audio Classifier, Audio Transcriber, Text-to-Speech | Existing `@lil-local/audio-mediapipe`; actual YAMNet Chromium inference proved, Parakeet and TTS APIs implemented but large-model real inference pending. |
| Universal Embedder + Semantic Retriever | **This implementation:** new `@lil-local/retrieval-mediapipe`, one shared model/session, semantic in-memory index, real text-model inference and package compatibility. |
| Decision Maker / Dino Game | Compare against separate `@lil-local/decision-onnx` when helpful; not automatically the same API/model or an application feature. No new unverified wrapper added. |
| Language detector, text classifier and text embedder | Existing encoder/classifier/moderation/text-model packages cover some goals but not necessarily the upstream SDK tasks. No duplicate installs without a motivating, qualified gap. |

We deliberately **did not copy** the entire gallery or assign the runtime
to ViiRM. In particular ViiRM is a character creator, not a webcam
tracking product.

## Implemented package boundary

`packages/retrieval-mediapipe/`: opt-in SDK
`@mediapipe/tasks-retrieval@1.2.2`, matching version-pinned WASM,
`BrowserSubstrateRunner` lifecycle, and native MediaPipe
`UniversalEmbedder` + `SemanticRetriever` created from its single
`getProvider()`. The in-memory store and text chunker are owned by
the same loaded runner and disposed on unload. Public API:

- `createUniversalRetrievalSubstrate({model, wasmBase?, chunkSize?,
  chunkOverlap?, l2Normalize?, ...})` needs an explicit model;
  no weights or runtime initialization happen on import.
- Inference requests: embed text, embed raw image bytes, embed normalized
  audio PCM; insert text/image/audio records; scored text query; list IDs,
  delete specific IDs and clear the volatile index.
- Returns the real native float-embedding heads (including dimensions)
  or the native scored `RetrievalResult`s. Nonfinite, missing or
  quantized-only float results and invalid/unbounded requests fail closed.
- The playground panel consumes the *package*, not a copied SDK implementation.
  It can select a caller-owned model File or explicit URL; one clickable
  **pinned text model preset** only populates the URL and does not fetch
  anything until Load. Indexed data is deliberately not persisted.
- The external consumer fixture independently packs and installs the
  package; no private workspace imports or hardcoded user paths.

## Real model qualification

Publisher: Hugging Face
`litert-community/embeddinggemma-2-text-270m-litert-lm`;
immutable revision `9be6e8b90982095dc05c2bd162e4b954ee4dbac7`.
Official file `embeddinggemma-2-text-270m.litertlm`:
**164,626,432 bytes**, SHA-256
`2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb`.
The actual downloaded bytes were hashed locally. **No model weights
were committed to Git.**

Real Chromium production-preview session:
- Linux Mint desktop, NVIDIA RTX 2070 SUPER; `navigator.gpu` available
  and an adapter returned. The native SDK was allowed to choose its GPU
  device; no low-level instrumentation independently proved GPU execution.
- An isolated Chrome profile selected the exact verified File and clicked
  Load in the Lil Local model lab. The runtime loaded, without an automatic
  pre-discovery model fetch.
- `embed-text` computed a **768-dimensional** real vector for the
  text query. Preview began `-0.063, -0.032, -0.029, -0.026, -0.057`.
- `insert-text` indexed `doc-1` containing
  `A small cat sitting in a cardboard box.`.
- Native `search` with `cat in a box` returned `doc-1`, score
  **0.839** (one observed run, not a benchmark).
- Unload returned the panel to Not loaded. The browser recorded **zero
  runtime errors** (no model or index fixture was mocked).
- Test: `scripts/mediapipe-retrieval-browser-smoke.py`, run with
  `pnpm test:retrieval:browser` after obtaining the exact model.

## Verification levels and remaining work

- Focused test path: 7 RED-first contract cases, now GREEN, with mocked
  SDK boundaries for one shared provider, abort, source fetch on explicit
  load, output validation and cleanup-on-construction-failure.
- Independent pack+TypeScript+Vite consumer compatibility covers
  **23 packages** including this one.
- Browser model verification **establishes text embeddings and text
  retrieval on this particular desktop**, not real multimodal image/audio
  inference. Testing the 388 MB Text-Vision and 485 MB Omnimodal official
  bundles is a separate qualification step, as is camera/audio input
  capture, long-term persistence and actual worker isolation.
- The model's SHA-256 was verified before running the local browser
  smoke. **The package's arbitrary URL load path does not independently
  hash streamed remote bytes**; do not treat a pinned catalog entry as
  automatic integrity validation in all consumers.
- MediaPipe may have its own usage/performance telemetry. This
  adapter does not explicitly upload embeddings/documents, but the SDK
  should not be advertised as fully offline-network-silent without an
  audit. Default WASM source uses pinned jsDelivr; self-hosting is possible.
- Authoritative `pnpm verify`: **exit 0** on the completed package,
  including full workspace typechecks/tests, 18 boundary tests, all 23
  independent packed-consumer packages, runtime qualification suite,
  conversion evidence validation and production builds. Existing Vite
  ineffective-import and large-chunk advisories remain nonfatal.
- No model release, push, dependency migration, or unrelated app
  modifications were made.
