# Bonsai 2 2B qualification — 2026-10-21

## Scope

Complete and preserve the pre-existing uncommitted Bonsai 2B playground
candidate without pretending that module bundling or shader-f16 implies
actual generation. Preserve the distinct Bonsai Image 4B adapter work.

## ONNX decoder: audited incompatible, not qualified

Immutable Hugging Face source:
`danyalmalik/Ternary-Bonsai-2-2B-ONNX-WebGPU`, revision
`a3c26c4d363186f744b20e08d585ce43e2ce7e8d`.

Downloaded **only the 1,054,038-byte decoder graph**, never the
503,981,056-byte decoder weight blob or 262,239,232-byte embedding weights.
Verified graph SHA-256:
`a4b3e8a9633cfffa35104ccbf2bd748c26f36cd8b1281c8443d47b82ecd0268f`.

Using Python ONNX 1.23.0, loaded the graph with
`onnx.load(..., load_external_data=False)` and recursively counted nested
graph nodes. **18 Scan nodes, 0 Loop nodes, 18 If nodes**, 4,706
nodes total. The logits output still includes a variable sequence axis
with vocabulary size 248,320. ONNX Runtime WebGPU does not implement Scan,
so these parts execute through CPU fallback and can cause pathological
token-by-token GPU round trips. Independent evidence from WebBrain's
Bonsai investigation agrees:
https://github.com/webbrain-one/webbrain/blob/main/docs/webgpu-models.md

The published ONNX decoder fails browser readiness regardless of
shader-f16 availability. The existing embedding-only probe remains as
**explicit opt-in**, not as proof that text generation works; invoking
the standalone script with no flags performs no browser/GPU/model work.

## Native route: concrete real test, NOT accepted yet

PrismML model: `prism-ml/Ternary-Bonsai-2-2B-gguf` revision
`683f7be2629342a50987627b40881e54761a2430`.
Downloaded only `Ternary-Bonsai-2-2B-PTQ1_0.gguf`
(429,183,456 bytes, SHA-256
`36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`),
not the demo's multi-GB default setup.
Official PrismML llama.cpp release:
`prism-b10787-73bfe6c`, Linux x64 CUDA 12.8 archive
(208,658,605 bytes, SHA-256
`f61ac818da2c8cddc47d75c955b6f5bf821738138760842dd50413c5181cdfe1`).
Hashes verified before extraction.

Isolated execution with `llama-cli`: `-ngl 0 -c 256 -t 2 -tb 2
-b 64 -ub 64 -n 8 --no-warmup --simple-io`, 75-second outer timeout and
low scheduling priority. **Exit 134**, GGML assertion:

`ggml.c:3413 GGML_ASSERT(ggml_is_padded_1d(a)) failed`

The source release function at that line is `ggml_scale_impl`.
No generated token, working live server, CUDA model inference or
quality/latency claim resulted. No driver lock was observed. Retain
the exact evidence and do not falsely classify this as working.

## Product behavior implemented

- Candidate remains on the model list for inspection, clearly marked blocked
  on the original browser ONNX graph. No fake Generate button on ONNX.
- Hardware adapter inspection remains opt-in and does not load model weights.
- Alternate **local native server** panel permits connection ONLY to
  `http://127.0.0.1:<port>` after a user starts a real PrismML native server.
- The client demands `/v1/models` advertise a Bonsai 2 2B model before
  enabling chat, validates the returned completion's model identity and
  non-empty actual generated text, bounds prompts and generation, and
  refuses automatic redirect to remote servers.
- Runtime connectivity and generation must still be demonstrated after the
  native backend mismatch above is corrected. Client tests use controlled
  fetch responses; they are **not runtime acceptance evidence**.

## Boundaries and continuation

No large ONNX weight downloads, GPU inference or driver changes.
Native archive/model remain in an isolated local cache, not Git.
The full source is a research candidate rather than a production model.
To qualify Bonsai 2: first resolve a PrismML-native loader compatible
with this exact 2B checkpoint, then capture real text output and
browser-to-native API integration, including clean model unload.
Alternatively audit/rewrite the ONNX decoder to remove Scan and verify
real WebGPU prefill/decode with bounded jobs and explicit memory estimates.

Do not force a model onto the RTX 2070 SUPER just because it is small.

## Real UI walkthrough — Chromium, GPU disabled

Local Vite server + disposable headless Chromium profile with
`--disable-gpu --disable-software-rasterizer`. Navigation from the model
catalog now follows the intended two-stage flow: select a model for details,
then use the explicit **Open** action in those details. The dedicated Bonsai
panel rendered the audited 18-Scan warning and opt-in local server connection.
No generated-content button appeared before native model identity validation.
**Zero page errors; zero external requests; no GPU inference.**
This is a UI-routing verification, NOT a model or native-server acceptance test.
