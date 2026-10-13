# D1 Omni ONNX/WebGPU text decision verification — 2026-10-12

## Scope

Real one-question **text** inference using the checked-in
`@litert-playground/decision-onnx` adapter on the authorized `yuzu-mint`
Linux Mint desktop. Vision and audio are out of scope, and there was no
integration into Veyax's autonomous sensing or action loop.

## Reproducible source

- Model `onnx-community/d1-omni-600M-ONNX` at
  `4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0`.
- `onnx/model_quantized.onnx` and its `.onnx_data` weights.
- `@huggingface/transformers` 3.8.1 tokenizer.
- `onnxruntime-web` 1.30.0 (WebGPU EP requested).
- Chromium, headless, Vulkan ANGLE (`--enable-unsafe-webgpu`,
  `--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE`,
  `--use-angle=vulkan`, `--disable-vulkan-surface`).

Chromium's `navigator.gpu.requestAdapter().info` identified
`vendor: nvidia` and `architecture: turing`. The default unmodified
headless configuration returned SwiftShader, so **these flags mattered**.

## Observed results

1. Model entry rendered and opened in Chromium; no JS page exceptions.
2. Before clicking Load, no Hugging Face model fetches occurred.
3. Actual pinned tokenizer loaded, with BOS 1 and reserved ids:
   mask=16; state=17; question=18; option=19; option_end=20; decide=21.
4. After explicit Load, `onnxruntime-web` created a session; UI reported
   `Ready (WebGPU provider requested)`. No failed network requests.
5. The default text request:

   - State: `I was charged twice for the same order. I need a refund.`
   - Question: `Which team should handle this?`
   - Criteria: `billing`, `technical`, `fraud`, with descriptions in
     `D1DecisionPanel.tsx`.

   **Observed answer:** billing 99.56%; technical 0.29%; fraud 0.15%.

   **Observed inference duration:** ~1011.4 ms (single invocation; includes
   wall time measured around session.run and browser-side encoding).

6. ONNX Runtime emitted warning:
   `Some nodes were not assigned to the preferred execution providers`.
   It explains shape-related nodes may run on CPU; the measurement establishes
   a working **WebGPU-requested** session on NVIDIA, not every op on GPU.
7. `SMOKE_PASS true`; no error panel.

## Other checks

- Seven new deterministic question-encoding/probability tests passed.
- Real pinned tokenizer marker IDs were independently inspected.
- Package and playground TypeScript checks passed.
- Chromium UI/no-auto-download smoke passed.
- Packed external consumer import/typecheck/build for the new
  `@litert-playground/decision-onnx` package passed.
- Final `pnpm verify` on the completed source tree exited **0** on
  2026-10-12: workspace typechecks/tests, dependency boundaries, packed
  consumer compatibility (including the new package), conversion checks,
  qualification suite, and production builds. The build emitted only the
  existing large-chunk advisory warning.

## Limits

One sample is not a latency distribution, accuracy evaluation, numerical
reference comparison, calibration, VRAM profiling or robust browser support
matrix. A genuine loaded browser session was exercised; there are no claims
for voice/vision ONNX graph composition, Veyax downstream consumption, or
session load abort/cancel. Chrome's HTTP caching of weights has not been
verified against Lil Tart's managed model library. Current session outputs
are advisory-only and may not authorize external actions.
