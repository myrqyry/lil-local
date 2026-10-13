<!-- meristem-template:v1 -->
# Feature Requests

Requested capabilities that are not currently implemented.

This is not a backlog priority order. Keep requests here when remembering the capability prevents repeated rediscovery.

Use entries like:

## Request

**Capability:**  
**Why it matters:**  
**Current gap:**  
**Constraints:**  
**Related project state / references:**

## 2026-10-13 — Evaluate browser GGUF/WebGPU as a separate opt-in runtime

**Capability under consideration:** A separately qualified browser GGUF inference lane for text and embeddings, alongside (not inside) the existing LiteRT and optional ONNX routes. Candidate: `paulrobello/webllm` (`@paulrobello/webllm`, MIT license).

**Why it matters:** Browser GGUF support could reuse some existing local GGUF assets without converting them to LiteRT. Its worker-hosted llama.cpp/ggml-WebGPU path, explicit model-weight budget, multi-tier embeddings, and versioned/fingerprinted conversation/KV snapshot approach are worth investigating where the currently qualified LiteRT/text models cannot serve a requested capability.

**Current overlap and non-goals:** Lil Tart already has an implemented, priority-aware `InferenceCoordinator` (FIFO equal-priority order, queued cancellation, telemetry and regression tests) and a proven LiteRT package boundary. Its D1 Omni Q8 ONNX path is deliberately opt-in. `webllm`'s `Scheduler`, `StreamRouter`, and `GameLoop` are explicitly experimental, behind an unstable `./internal` API, and **not wired into its live inference loop**; do not import them as if shipping hierarchical, frame-budget-aware scheduling. No general runtime refactor, Bun migration, or premature public API.

**Source-grounded limitations (checked 2026-10-13):** GGUF only; patched llama.cpp `ggml-webgpu` WASM binaries must be hosted, with a special memory64 build for files larger than 3.5 GiB. Browser regression coverage focuses on Chrome, JSPI, and a 128 MiB storage-buffer binding limit. Its model-support guide reports that MoE expert counts are parsed but **not executed**, so MoE models can yield incorrect dense-forward output. Unsupported architectures or chat templates may silently produce incorrect outputs or omit tool schemas. The default 8 GiB `memoryBudget` is an allocation limit tracked by that engine, **not a query of total available VRAM** or a guarantee that the GPU can accept another model.

**Evidence required before a dependency:** Confirm published package and actual bundled WASM/asset requirements; verify licensing per model separately; choose one small supported dense GGUF and an independently specified prompt/expected output. Test in an isolated browser harness on the real 8 GiB NVIDIA/Turing device with the actual browser flags/backend. Record model SHA, architecture, quantization, actual GPU adapter/device limits, download and load paths, cold-start, prefill, TTFT, decode tokens/s, live memory pressure, cancel/dispose, typing/interaction responsiveness, multi-turn and worker crash/error behavior. Compare against a *relevant* Lil Tart text model without implying model-quality equivalence. Reject unsupported architecture, MoE, and tool templates before claiming success. Keep explicit opt-in and source-backed runtime receipts; no automatic downloads on page navigation.

**Status:** Repository/architecture review only; no dependency, WASM, GGUF weights, model inference, runtime qualification, or app integration performed.

**Upstream:** https://github.com/paulrobello/webllm ; https://github.com/paulrobello/webllm/blob/main/docs/MODEL_SUPPORT.md ; https://github.com/paulrobello/webllm/blob/main/docs/BENCHMARKS.md .
