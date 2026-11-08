# Browser inference substrates

Lil Local is a local-first **web inference qualification lab**, not a promise that every model format runs through LiteRT. A *substrate* is the model execution engine/loader; an *execution backend* is where that engine requests computation (WebGPU, WebNN, or WASM). Neither is the format (GGUF, ONNX, TFLite) or the model's task ABI.

## What exists now

| Path | Current scope | Shared substrate runner? |
| --- | --- | --- |
| LiteRT.js | Existing managed model packages, WebGPU/WebNN/WASM selection, caching, preflight, scheduler, telemetry | Existing dedicated runtime remains unchanged, not migrated |
| ONNX Runtime Web | Pinned D1 Omni 600M Q8 text-only decision path, explicitly WebGPU-requested | **Yes**, via `d1OnnxSubstrate` in the playground |
| ONNX Runtime Web / Apple SHARP | Opt-in raw-predictor single-image → 3D Gaussian PLY. Model weights not bundled; hardware execution unqualified. | **Yes**, via public `@lil-local/sharp-onnx` / `createSharpSubstrate` |
| Transformers.js / ONNX Depth Anything V2 Small | Revision-pinned Q8/WASM depth with actual Chromium inference and a normalized `DepthFrame`; FP16/WebGPU selection exists but is not yet runtime-qualified. | **Yes**, via public `@lil-local/depth-estimation` / `createDepthAnythingV2Substrate` |
| MediaPipe Tasks Vision | Nine opt-in frame processors including SHA-verified Selfie foreground mask; independent still-image MagicTouch session. | **Yes**, via public `@lil-local/vision-mediapipe` |
| LiteRT-LM Web | Opt-in text-only EmbeddingGemma 2 via `embedding-litertlm` (browser model execution not yet qualified) | **Yes**, as an external reusable package |
| Other Transformers.js and pipeline-owned engines | Existing task-specific packages without a shared browser substrate wrapper | Not generally migrated |
| GGUF/WebGPU (`paulrobello/webllm`) | Reviewed candidate, not downloaded or qualified | Not implemented |
| Other browser engines and custom WGSL | Future candidates | Not implemented |

## The reusable contract

`@lil-local/inference-core` exports `BrowserSubstrate`, `LoadedBrowserSubstrate`, and `BrowserSubstrateRunner` through the public entrypoint. The substrate declares its ID, model format, capabilities, requested execution backends, and an explicit `create()` loader. The loaded implementation exposes `infer(input)` and `dispose()`.

The runner never downloads weights or starts a GPU model in its constructor. `load()` is explicit, inference is single-flight, disposal waits for active inference, and models finishing a load after disposal are released instead of becoming ready. Failures propagate. `load(onProgress, signal)` forwards an optional `AbortSignal` to `create(onProgress, signal)`; a pre-aborted signal prevents loading, and a signal aborted during loading suppresses late progress and prevents a returned model from becoming ready. That returned model is disposed **before** cancellation settles, and cleanup failures remain errors rather than unearned success claims. An unsuccessful late release locks the runner in `failed` (not `idle`) so it cannot allocate another session on top of possibly leaked resources; `dispose()` still reports the original cleanup error. This is **cooperative cancellation**, not a claim that arbitrary runtime/network/compilation operations are interruptible. A loader that ignores the signal may continue consuming resources until its pending call returns. The runner does **not** silently fall back, own a global GPU memory budget, normalize model-specific tensor ABIs, or promote verification records. Its metadata describes **requested** backends, not proof that every operation ran on the GPU.

D1's loaded ONNX implementation also deduplicates direct session disposal: multiple callers await the same release promise and observe the same failure if GPU resource cleanup fails. If both ONNX loading/ABI validation and session release fail, its error retains both causes rather than silently replacing the original failure. These are deterministic lifecycle guarantees backed by unit tests, not proof of any particular GPU driver's release behavior.

The D1 ONNX panel uses this contract to drive the real `D1OnnxWebGpu.load()`, `decide()`, and `dispose()` calls. Its **Stop loading** control now aborts cooperatively and waits for the runner's cleanup; the D1 loader checks cancellation before tokenizer/graph stages and after the ONNX session returns, releasing any late-created session. `onnxruntime-web` is not passed a network cancellation signal by this path: Stop can therefore remain pending during an in-progress ORT download or graph compilation. A simultaneous loader failure is reported separately rather than mislabeled as a clean cancellation. The previous real D1 inference evidence lives in `docs/verification/2026-10-12-d1-webgpu-text.md`; cancellation unit tests and a no-weights browser UI smoke test do **not** constitute a fresh full model inference qualification.

## Public TypeScript lifecycle reference

The source of truth is [`packages/inference-core/src/substrate.ts`](../packages/inference-core/src/substrate.ts), exported from `@lil-local/inference-core`. This is a TypeScript interface, **not an HTTP endpoint**; an endpoint-only API checker cannot score its coverage.

| Member | Public contract | Constraint / error behavior |
| --- | --- | --- |
| `BrowserSubstrate<I, O, Progress>` | `id`, `format`, `capabilities`, `requestedBackends`, `create(onProgress?, signal?)` | Nonempty metadata required; requested backends are not observed backend evidence. |
| `LoadedBrowserSubstrate<I, O>` | `infer(input): Promise<O>`, `dispose(): Promise<void>` | Model-specific I/O and underlying cleanup belong to the adapter. |
| `new BrowserSubstrateRunner(substrate)` | Own one session | Constructor does not load weights; invalid descriptors throw. |
| `runner.status` | `idle`, `loading`, `ready`, `running`, `failed`, `disposing`, `disposed` | Lifecycle status is not a GPU or model-parity receipt. |
| `runner.load(onProgress?, signal?)` | Explicit creation and optional cooperative cancellation | Only allowed in `idle`. Pre-aborted signals reject with `AbortError` before creation; late returned models are disposed before abortion settles. Failed late cleanup prevents reuse. |
| `runner.run(input)` | Await one inference result | Requires `ready`; concurrent runs are rejected. |
| `runner.dispose()` | Wait for pending load/run, release model | Repeated callers share pending disposal. Cleanup failures reject; final `disposed` status alone does **not** prove successful underlying release. |

The runner does not guarantee interruption of a loader that ignores `AbortSignal`, nor that computation actually ran on a requested backend. Runtime evidence must be established separately.

## LiteRT-LM JS / WASM compatibility

The **current** `@lil-local/text-gen` and EmbeddingGemma 2 paths pin
`@litert-lm/core@0.19.0`. The text pipeline, its worker, and the
embedding substrate explicitly call `getOrLoadGlobalLiteRtLm()` with
matching **0.19.0** WASM assets before constructing a native engine.
Consumers may supply their own `wasmBase` for an offline/CSP-controlled
origin; the default is the version-pinned jsDelivr path. Custom hosting
must include the complete same-version WASM/JS variant set. A source-level
pin and successful typecheck are **not** proof of model generation on a
specific browser/device.

Historical context: the earlier `0.18.1` integration had to override an
upstream default pointing at `@0.12.1/wasm`. A Chromium smoke subsequently
loaded the **real 0.18.1 WASM module** but no model weights or generated text;
that remains an asset-alignment receipt, not model inference. See
`docs/verification/2026-10-18-litertlm-wasm-alignment.md`. The current
runtime version is `0.19.0`, not `0.18.1`.

## Recent shared model wrappers

The [Live Vision model receipt](verification/2026-11-07-controlnet-models.md)
qualifies a *real inference path* for the SHA-pinned Selfie Segmenter and
Depth Anything V2 Q8/WASM in Chromium, using the existing playground's
camera/preview lifecycle. Depth Anything V2's ONNX Runtime `.mjs` and
`.wasm` are copied from the installed dependency and self-hosted from
`/ort/`; pinned model weights are still fetched opt-in. A recorded **2,636 ms**
CPU depth frame and unqualified FP16/WebGPU path do not support a real-time
claim. The package exposes normalized relative depth; the browser UI creates
its own grayscale conditioning preview. See
[depth package usage](../packages/depth-estimation/README.md).

## Adding a new runtime

A new engine needs a tested, real runtime-specific adapter and task contract (including input preprocessing and output decoding), not merely a new descriptor or model card. Preserve source/revision and model/weight integrity, actual execution-provider evidence, explicit failure/fallback receipts, and separated durable-vs-session verification. Test actual inference with independent expected outputs and failures, model load and inference performance, cleanup, cancellation, worker/device failures, and memory pressure on target hardware.

The published entrypoint must also pass Lil Local's packed external-consumer compatibility tests; `pnpm verify` remains the qualification gate. A shared JS interface is only structural evidence, never proof of model correctness or browser compatibility. On an 8 GiB GPU, an engine's own weight budget cannot prove the global GPU's available memory.

The future direction is **multiple separately qualified runtime families**, not forcing WebNN, WebGPU, WASM, ONNX, GGUF, and LiteRT into one fictitious runtime or one global backend selector.
