# @lil-local/runtime-litert

Shared managed LiteRT.js runtime for browser inference consumers.

This package owns generic runtime concerns. Product-specific concepts belong in consumer apps such as PodQast or Live Streamer Gemini.

## What it provides

- Runtime capability probing for WebGPU, WebNN, and WASM.
- Automatic backend order: WebGPU → WebNN → WASM.
- Explicit backend requests that fail rather than silently changing hardware.
- Per-model backend compatibility filtering.
- Compiled-model caching and concurrent-load deduplication.
- Shared inference serialization through `InferenceCoordinator`.
- Named-signature and ordinary model inference.
- Model preflight with input/output inspection and bounded synthetic-input generation.
- Compile, inference, fallback, output-count, and tensor-copy telemetry.
- Bounded telemetry history for long-running apps.
- Cancellation and stable `InferenceError` categories.
- Model-level and runtime-level disposal.
- NPY loading and asset-buffer access through the shared `AssetResolver` contract.

## Basic usage

```ts
import { createHttpAssetResolver } from '@lil-local/inference-core'
import { createLiteRtRuntime } from '@lil-local/runtime-litert'

const context = await createLiteRtRuntime({
  backend: 'auto',
  assets: createHttpAssetResolver(document.baseURI),
})

await context.liteRt.loadModel('/models/model.tflite')

const preflight = await context.liteRt.preflight('/models/model.tflite')
console.log(preflight.resolvedBackend, preflight.inferenceDurationMs)
```

## Public managed-runtime API

The authoritative declarations are [`src/types.ts`](src/types.ts), exported through [`src/index.ts`](src/index.ts). This is a **TypeScript interface**, not an HTTP API; endpoint-only documentation checkers cannot establish coverage here.

| Method | Contract | Important boundary |
| --- | --- | --- |
| `loadModel(path, options?)` | `Promise<CompiledModel>` | Loads/compiles explicitly; options include `accelerator`, `supportedBackends`, `signal`, `onProgress`, and `webNNOptions`. |
| `predict(path, input, options?)` | `Promise<LiteRtModelOutput>` | Input is one Tensor, a Tensor array, or a named Tensor map. The returned output is a Tensor array or named map. |
| `predictWithSignature(path, signature, input, options?)` | `Promise<LiteRtModelOutput>` | Runs an explicitly named model signature. |
| `preflight(path, options?)` | `Promise<LiteRtPreflightResult>` | Performs real synthetic inference; `signature`, `maxTensorElements`, and `createInputs` may be supplied. |
| `getModelInfo(path, options?)` / `getDiagnostics(path, options?)` | Model metadata or diagnostics, possibly `undefined` | Metadata is not independent inference-parity evidence. |
| `getTelemetry()` / `clearTelemetry()` | Read bounded telemetry / clear it | Telemetry is session history, not durable qualification. |
| `loadNpy(path, signal?)` / `fetchBuffer(path, signal?)` | `Promise<Float32Array>` / `Promise<ArrayBuffer>` | Asset access follows the configured resolver. |
| `createTensor(data, shape)` / `readTensor(tensor)` | Tensor construction / typed-array readback | Typed inputs: float32, int32, int8, uint8. |
| `supportsGpuBufferTensors()` | `boolean` | Capability observation, not a guarantee every model operation ran on GPU. |
| `disposeModel(path)` / `dispose()` | `void` | Release model-specific resources / whole managed runtime. |

The `LiteRtModelOptions.inferencePriority` field controls queue priority for inference/preflight, **not model identity or loading order**. The runtime-wide `backend` preference and model-specific `accelerator` are separate choices. See the source and tests for precise errors and disposal behavior.

## Backend semantics

`backend: 'auto'` tries usable model backends in this order:

1. WebGPU
2. WebNN
3. WASM

A compile failure advances to the next usable backend. The resulting `LiteRtModelInfo` records the requested backend, resolved backend, compile duration, and fallback count.

An explicit request such as `accelerator: 'webgpu'` does **not** fall back. This makes tests, benchmarks, and user-selected hardware truthful.

Per-model restrictions can be supplied with `supportedBackends`.

## Preflight

`preflight()` compiles the model, reads input/output metadata, creates bounded zero-valued inputs for fixed `float32`, `int32`, `int8`, and `uint8` tensors, executes one inference, and returns a `LiteRtPreflightResult`.

Dynamic or unusually large inputs should provide `createInputs` instead of relying on generated inputs.

Preflight is intentionally explicit. Consumers should not automatically execute potentially expensive synthetic preflight inference merely by listing a model.

## Coordination

All managed `predict`, `predictWithSignature`, and `preflight` calls use the shared `InferenceCoordinator` unless a consumer supplies a different coordinator.

The coordinator serializes hardware inference, supports aborting queued work, and emits queue/start/finish events. This prevents unrelated browser models from casually fighting over accelerator ownership, a surprisingly popular hobby among otherwise respectable neural networks.

## Telemetry

```ts
const records = context.liteRt.getTelemetry()
```

Records distinguish `compile`, `inference`, and `preflight` events and include:

- model path
- requested and resolved backend
- compile duration
- inference duration when relevant
- fallback count
- input/output count
- tensor-copy count

History is bounded (`telemetryLimit`, default 512) so persistent apps do not grow an immortal diagnostics array.

## Consumer boundary

Good shared-runtime responsibilities:

- backend selection
- model loading and caching
- tensor/runtime helpers
- preflight
- resource coordination
- telemetry and error normalization

Keep these in the consuming product instead:

- episode/cast semantics
- streamer or OBS actions
- game-agent beliefs and recovery logic
- product-specific buffering and UI
- model-specific preprocessing unless it is part of a reusable model package

The practical rule is: if the code still makes sense without any particular product existing, it is a candidate for this package or another shared package.
