# Packed-consumer LiteRT WebGPU numerical qualification attempt — 2026-11-01

## Intent and implementation

Extend the existing real packed-consumer numerical probe with a **strict**
WebGPU lane using the same pinned 708-byte `add_10x10.tflite` asset,
two distinct inputs, independent output oracle, telemetry and cache checks.

```bash
pnpm test:compatibility:webgpu
```

This is an opt-in hardware-dependent check, not part of routine `pnpm verify`.
It requests `backend: 'webgpu'` with no fallback, validates the reported
resolved backend, rejects SwiftShader/unknown adapter identities, and checks
all returned values against the arithmetic oracle.

## Observed outcome

**Not qualified.** On `yuzu-mint`, Chromium **156.0.8078.47** with Vulkan
ANGLE / unsafe-WebGPU flags, the browser reported **no WebGPU adapter**.
The probe terminated with a nonzero exit **before** model compilation or
inference. The pre-existing `pnpm probe:webgpu-matmul --size 64`, which uses
the same flags, independently failed with `No WebGPU adapter`.

The native system *does* expose Vulkan NVIDIA GeForce RTX 2070 SUPER
with NVIDIA driver **595.99.02** (`vulkaninfo --summary`), and
`nvidia-smi` sees the GPU. That does not establish browser GPU readiness.
The mismatch between native Vulkan availability and browser adapter
discovery is unresolved; do not mark the WebGPU model as passing.

### Chrome Beta comparison

On the same host, **Google Chrome Beta 158.0.8127.2** also returned
`WebGPU adapter unavailable` in the strict packed-consumer probe with
Vulkan/ANGLE flags. The existing matmul probe independently returned
`No WebGPU adapter` using those flags.

An isolated headless Chrome Beta adapter test against localhost found:

- Default flags: `navigator.gpu` exists, `requestAdapter()` returns null.
- `--enable-unsafe-webgpu` alone: an adapter exists, but reports
  `vendor=google`, `architecture=swiftshader` (**software**, not NVIDIA).
- `--enable-unsafe-webgpu` plus Vulkan/ANGLE flags:
  `requestAdapter()` returns null.

None of these tests establish hardware GPU inference. The strict probe
correctly refuses SwiftShader as evidence of NVIDIA execution.

## Boundaries

WASM numerical parity is independently verified and remains a separate
qualified path. This record does not claim WebGPU inference, numerical
correctness or the physical GPU's execution. Recheck browser adapter
availability and driver/Chromium compatibility before assigning any
WebGPU status. Do not accept SwiftShader software execution as proof
of NVIDIA hardware execution.
