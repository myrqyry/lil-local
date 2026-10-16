# GPU Goldmines: Lil Tart WebGPU feasibility (2026-10-16)

Research: https://arxiv.org/abs/2610.17321

This is a completed, bounded hardware experiment, **not** an inference
speedup. The command is opt-in; production model loading is untouched.

## Runtime boundary

Lil Tart currently uses LiteRT.js 2.5.3 and an opt-in ONNX Runtime Web
1.30.0 path. Inspection of their installed APIs and Lil Tart's source
found:

- packages/runtime-litert/src/context.ts selects an accelerator
  (WebGPU/WASM/WebNN) for LiteRT loadAndCompile. It cannot replace
  LiteRT's matmul or FlashAttention kernels.
- packages/decision-onnx/src/runtime.ts creates a WebGPU-requested
  ONNX session. Its provider supports layout and profiling options,
  but not substitution of the paper's custom kernels.
- No inference implementation currently consumes custom WGSL matmul,
  or the result of this benchmark. A standalone matmul result cannot
  improve existing model inference unless this changes.

Consequently no automatic runtime switching, speculative profile
database or unproven acceleration claims were added.

## Real on-device experiment

Command: pnpm probe:webgpu-matmul

On Linux Mint / Chromium WebGPU Vulkan / NVIDIA Turing RTX 2070 Super,
the probe ran native browser WGSL matrix multiplication using 8x8
and 16x16 workgroups. The browser exposed timestamp-query, subgroups,
and experimental subgroup matrix support. The independent 64-element
compute diagnostic also produced 64/64 expected outputs.

The probe:
- Uses a secure local loopback page, not an inaccessible blank origin.
- Compiles both variants and actually dispatches work to the GPU.
- Checks **every output element** against a CPU floating-point reference.
- Excludes warm-up, then measures six rounds of eight dispatches each.
- Refuses to select a configuration unless its median is at least 10%
  faster AND its sample range does not overlap the other.
- Fails closed on missing WebGPU, shader errors or numerical errors.

Shape 256 x 256 x 256 (FP32):

| Workgroup | Median wall ms / dispatch | Maximum absolute output error |
|---|---:|---:|
| 8 x 8 | ~0.275 | 0.00000140 |
| 16 x 16 | ~0.275 | 0.00000140 |

A second 128 x 128 x 128 run found both at ~0.2875 ms per dispatch.
At 512 x 512 x 512, the observed medians were approximately 0.881 ms
(8 x 8) and 0.669 ms (16 x 16); however, the six timing ranges
overlapped, so this run also refused to select a reliable winner.
Both passed complete CPU numerical comparison with a maximum absolute
error of 0.00000319.

Decision in all three sizes: selected=null, reason=indistinguishable.
This does not imply that 8 x 8 and 16 x 16 are equally efficient.
It means these specific noisy wall-time measurements do not establish
a sufficiently reliable choice under our conservative criterion.

Measured wall time includes JavaScript submission, GPU queue and buffer
readback amortized over eight dispatches. It is NOT an isolated GPU
timestamp measurement, model prefill/decode benchmark or verified model
latency improvement. Results may change across browsers, devices,
shapes and workload types.

An initial secure Chromium launch failed to acquire an adapter;
a subsequent direct WebGPU capability check and probe both passed.
The tool reports adapter errors instead of silently using fake data.

## Reproduction

    pnpm test:webgpu-kernel-probe
    pnpm probe:webgpu-matmul
    pnpm probe:webgpu-matmul --size 128
    CHROMIUM_BINARY=/path/to/chromium pnpm probe:webgpu-matmul

The tool emits JSON with device identity, raw timing samples, complete
correctness verdicts, the conservative selection decision and an
explicit inferenceIntegration=none boundary. No model downloads.

## Real optimization gate

Before using kernel tuning in an inference path, a runtime must expose
custom kernels or Lil Tart must own that GPU compute path. Then test
real model shapes, multiple repeatable inference requests, numerical
output parity, and separate prefill/decode latency before selecting
anything. The existing LiteRT/ONNX backends stay unchanged for now.
