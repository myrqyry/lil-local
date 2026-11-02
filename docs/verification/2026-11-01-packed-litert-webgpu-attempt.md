# Packed-consumer LiteRT WebGPU numerical qualification — 2026-11-01

## Intent and implementation

Qualify the **actual packed** `@lil-local/runtime-litert` WebGPU path using
the pinned 708-byte Google `add_10x10.tflite` fixture, two distinct inputs,
an independent arithmetic oracle, strict backend identity checks, telemetry
and compiled-model cache verification.

```bash
DISPLAY=:0 \
CHROMIUM_BINARY=/usr/bin/google-chrome-beta \
PACKED_BROWSER_HEADED=1 \
pnpm test:compatibility:webgpu
```

This remains an opt-in hardware/browser-dependent check rather than part of
routine `pnpm verify`. It requests `backend: 'webgpu'` explicitly with no
backend fallback and rejects missing, unknown, or SwiftShader adapters.

## Initial headless failure

The first Chromium 156 and Chrome Beta 158 headless attempts did **not**
qualify hardware inference. With forced Vulkan/ANGLE flags,
`navigator.gpu.requestAdapter()` returned null. With
`--enable-unsafe-webgpu` alone, Chrome Beta exposed only
`vendor=google`, `architecture=swiftshader`, which the probe correctly
rejected as software execution.

Native `vulkaninfo` and `nvidia-smi` already saw the RTX 2070 SUPER, so
those failures were kept as browser-harness evidence rather than treated as a
model or driver failure.

## Headed Chrome Beta qualification

A headed X11 Chrome Beta session using the browser's working feature shape
(`--ozone-platform=x11`, experimental web-platform features,
WebGPU developer features, and `Vulkan`) returned a real hardware adapter:

```text
vendor=nvidia
architecture=turing
device=0x1e84
description=NVIDIA GeForce RTX 2070 SUPER
```

The first hardware inference then exposed a separate Lil Local bug:
`readTensor()` called LiteRT.js `toTypedArray()` directly on a WebGPU
resident output tensor. LiteRT.js correctly rejected that with
`Cannot convert a Tensor with WebGPU memory to a TypedArray.`

Lil Local now keeps synchronous `readTensor()` for host-memory tensors and
adds `readTensorAsync()`, which delegates to LiteRT.js `Tensor.data()`.
That upstream path copies accelerator memory back to host memory when needed
before returning the typed array.

## Recorded passing evidence

- Browser: Google Chrome Beta **158.0.8127.2**, headed on X11.
- Hardware adapter: **NVIDIA GeForce RTX 2070 SUPER**, Turing, device
  `0x1e84`.
- Model: Google `add_10x10.tflite`, **708 bytes**.
- Verified model SHA-256:
  `1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2`.
- Public runtime sequence:
  `createLiteRtRuntime({backend:'webgpu'})`, `loadModel`,
  `createTensor`, `predict`, `readTensorAsync`, `dispose`.
- Input case 1: `3 + 5` → every output element `8`.
- Input case 2: `-2 + 7` → every output element `5`.
- Maximum absolute numerical error across both cases: **0**.
- Resolved backend reported by the managed runtime: **webgpu**.
- Compiled-model cache reuse: **true**.
- Telemetry: exactly one `compile` and two `inference` records.
- Packed-consumer install, typecheck, contract tests and build: **pass**.
- Probe exit status: **0**.

## Evidence boundary

This qualifies one small fixed-shape model through Lil Local's packed managed
LiteRT WebGPU API on one Chrome Beta / NVIDIA configuration. The hardware
adapter identity, explicit `webgpu` backend resolution and GPU-resident
output establish the real GPU execution path for this test.

It does **not** establish numerical correctness for arbitrary models, prove
that every operation in every future model will be GPU delegated, qualify
headless Chrome, Android, WebNN, other GPUs/browsers, or fully offline use.
WASM numerical qualification remains independent.
