# Real Transformers.js WebGPU load / generation / disposal acceptance — 2026-10-20

**Scope:** qualify the already-committed 144cb8d Lil Local TransformersTextPipeline with a real installed model and NVIDIA GPU. This is an external acceptance experiment, not a code change or mock test.

## Setup

- Local host: NVIDIA GeForce RTX 2070 SUPER (Turing, 8 GiB), Chromium under isolated Xvfb, existing test-only Chromium model-cache profile. An already-running local app on 127.0.0.1:5488 supplied the previously qualified origin/cache; a temporary Vite server on port 5563 served the actual packages/text-gen/src/transformers-pipeline.ts through native browser dynamic import. Both test Chromium sessions and the temporary Vite server were stopped.
- Model: onnx-community/Qwen3-0.6B-ONNX, revision 58ca03a5e7b354b480b1a82b34bde8dc7b854cc0, device webgpu, dtype q4. The browser's WebGPU adapter identified NVIDIA / Turing. The weight cache was already populated; this was warm cache, not first-download/offline qualification.
- Exercised: concurrent loadForDevice('webgpu') calls via Promise.all, a real run() on a short request, then two sequential dispose() calls. The model actually produced text; status ended disposed; no page errors or wrapper exceptions. Two browser runs completed; the second added GPU measurements while the browser was still alive.
- Generated text began with an internal thinking prefix and did **not** follow the short instruction to say only one word. This proves real inference executed, not instruction-following quality or suitability for a streamer.

## System-wide NVIDIA memory snapshots (MiB, second run)

| Moment | nvidia-smi memory.used |
| --- | ---: |
| Before browser launch | 4269 |
| Browser open, before model load | 4342 |
| After model load | 5826 |
| After generation | 5904 |
| Immediately following successful dispose() | 4685 |
| 1.4 seconds after disposal, browser **still open** | 4325 |
| After closing browser | 4268 |

**Inference justified:** live model disposal was followed by a decrease from 5904 MiB to 4325 MiB *without closing the test browser*. The after-disposal value is close to the browser-before-load snapshot. This is materially stronger than observing a decline only after browser exit.

**Limits:** nvidia-smi reports total GPU usage across processes, not per-model allocations; other running programs could affect deltas. This does not establish per-buffer release, deterministic timings, a leak-free application under long sessions, or stable VRAM on every device. Browser GPU/process accounting and asynchronous resource release may explain latency between snapshots. The warm-cache experiment did not test cold downloading, cache correctness, simultaneous long-running generations, or real cancellation during GPU execution.

## Other qualification boundaries

The mocked lifecycle regression tests and package TypeScript checks passed in commit 144cb8d (91 tests), and those remain distinct evidence tiers. Prior full pnpm verify reached playground build, then failed resolving onnxruntime-web/webgpu in concurrent uncommitted Bonsai2 playground work. That work was not modified or included in this acceptance test. No push, downstream consumer upgrade, cache reset or production app configuration change was performed.

Disposable harness and run logs remained only under local /tmp/lil-local-transformers-gpu-lifecycle-20261020*; these are not permanent artifacts or part of this commit.
