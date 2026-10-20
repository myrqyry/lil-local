# Qwen3 0.6B browser WebGPU fallback, October 19, 2026

Creator's Insight Engine attempted a real verified 614,236,160-byte Qwen3
LiteRT-LM checkpoint with NVIDIA Turing WebGPU. Lil Local's generic-to-native
backend fix advanced the initial failure, but installed upstream
`@litert-lm/core@0.18.1` then failed with
`wasmEngineSettings.delete is not a function`. Live examination found
that the WASM class lacks several methods demanded by the matching JS,
including `EngineSettings.delete` and `Engine.createStreaming`.
No LiteRT-LM Qwen inference succeeded; do not claim it has.

An independent real Chromium session on the same NVIDIA/Turing adapter
successfully loaded `onnx-community/Qwen3-0.6B-ONNX`, pinned Hugging Face
revision `58ca03a5e7b354b480b1a82b34bde8dc7b854cc0`, with
Transformers.js 4.2.0, `device:'webgpu'`, `dtype:'q4'`. The
generated_text result was an actual assistant response, including a
`<think>` prefix, after around 52 seconds total including first-run
weight downloads. This is real browser-model inference, but the short
token budget produced only an unfinished thinking channel—not yet a
quality-qualified summary.

Lil Local's TransformersTextPipeline previously hardcoded WASM, ignored
the configured model/dtype, and expected a non-standard `{output:[...]}`
result, which cannot parse Transformers.js v4's real
`[{generated_text:[{role,content}]}]` shape. Now accepts explicit
model/device/dtype/revision and a direct `loadForDevice` for downstream
consumers without fabricated LiteRT runtime context, and extracts the
assistant message from actual pipeline output. The scoped package uses
Transformers.js ^4.2.0; other package consumers remain independently
pinned to their earlier major version as needed.

Verification: 3 focused new tests failed before changes, followed by
83/83 text-gen tests and a full monorepo `pnpm verify` pass, including
external consumer compatibility. Live end-to-end inference through the
post-change Lil Local wrapper must still be independently tested.
