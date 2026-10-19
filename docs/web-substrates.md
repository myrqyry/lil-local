# Browser inference substrates

Lil Tart is a local-first **web inference qualification lab**, not a promise that every model format runs through LiteRT. A *substrate* is the model execution engine/loader; an *execution backend* is where that engine requests computation (WebGPU, WebNN, or WASM). Neither is the format (GGUF, ONNX, TFLite) or the model's task ABI.

## What exists now

| Path | Current scope | Shared substrate runner? |
| --- | --- | --- |
| LiteRT.js | Existing managed model packages, WebGPU/WebNN/WASM selection, caching, preflight, scheduler, telemetry | Existing dedicated runtime remains unchanged, not migrated |
| ONNX Runtime Web | Pinned D1 Omni 600M Q8 text-only decision path, explicitly WebGPU-requested | **Yes**, via `d1OnnxSubstrate` in the playground |
| Transformers.js and other pipeline-owned engines | Existing task-specific packages | Not migrated |
| GGUF/WebGPU (`paulrobello/webllm`) | Reviewed candidate, not downloaded or qualified | Not implemented |
| Other browser engines and custom WGSL | Future candidates | Not implemented |

## The reusable contract

`@litert-playground/inference-core` exports `BrowserSubstrate`, `LoadedBrowserSubstrate`, and `BrowserSubstrateRunner` through the public entrypoint. The substrate declares its ID, model format, capabilities, requested execution backends, and an explicit `create()` loader. The loaded implementation exposes `infer(input)` and `dispose()`.

The runner never downloads weights or starts a GPU model in its constructor. `load()` is explicit, inference is single-flight, disposal waits for active inference, and models finishing a load after disposal are released instead of becoming ready. Failures propagate. `load(onProgress, signal)` forwards an optional `AbortSignal` to `create(onProgress, signal)`; a pre-aborted signal prevents loading, and a signal aborted during loading suppresses late progress and prevents a returned model from becoming ready. That returned model is disposed **before** cancellation settles, and cleanup failures remain errors rather than fake successes. An unsuccessful late release locks the runner in `failed` (not `idle`) so it cannot allocate another session on top of possibly leaked resources; `dispose()` still reports the original cleanup error. This is **cooperative cancellation**, not a claim that arbitrary runtime/network/compilation operations are interruptible. A loader that ignores the signal may continue consuming resources until its pending call returns. The runner does **not** silently fall back, own a global GPU memory budget, normalize model-specific tensor ABIs, or promote verification records. Its metadata describes **requested** backends, not proof that every operation ran on the GPU.

D1's loaded ONNX implementation also deduplicates direct session disposal: multiple callers await the same release promise and observe the same failure if GPU resource cleanup fails. If both ONNX loading/ABI validation and session release fail, its error retains both causes rather than silently replacing the original failure. These are deterministic lifecycle guarantees backed by unit tests, not proof of any particular GPU driver's release behavior.

The D1 ONNX panel uses this contract to drive the real `D1OnnxWebGpu.load()`, `decide()`, and `dispose()` calls. Its **Stop loading** control now aborts cooperatively and waits for the runner's cleanup; the D1 loader checks cancellation before tokenizer/graph stages and after the ONNX session returns, releasing any late-created session. `onnxruntime-web` is not passed a network cancellation signal by this path: Stop can therefore remain pending during an in-progress ORT download or graph compilation. A simultaneous loader failure is reported separately rather than mislabeled as a clean cancellation. The previous real D1 inference evidence lives in `docs/verification/2026-10-12-d1-webgpu-text.md`; cancellation unit tests and a no-weights browser UI smoke test do **not** constitute a fresh full model inference qualification.

## Adding a new runtime

A new engine needs a tested, real runtime-specific adapter and task contract (including input preprocessing and output decoding), not merely a new descriptor or model card. Preserve source/revision and model/weight integrity, actual execution-provider evidence, explicit failure/fallback receipts, and separated durable-vs-session verification. Test actual inference with independent expected outputs and failures, model load and inference performance, cleanup, cancellation, worker/device failures, and memory pressure on target hardware.

The published entrypoint must also pass Lil Tart's packed external-consumer compatibility tests; `pnpm verify` remains the qualification gate. A shared JS interface is only structural evidence, never proof of model correctness or browser compatibility. On an 8 GiB GPU, an engine's own weight budget cannot prove the global GPU's available memory.

The future direction is **multiple separately qualified runtime families**, not forcing WebNN, WebGPU, WASM, ONNX, GGUF, and LiteRT into one fictitious runtime or one global backend selector.
