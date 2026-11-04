# MediaPipe Parakeet browser ASR — 2026-11-04

## Purpose and entry point
An **opt-in** second browser ASR path in Lil Local, alongside the existing Nemotron ONNX INT4 path. It uses Google's new `@mediapipe/tasks-audio@1.2.2` `AudioTranscriber` without inventing a model runner or interfering with `@lil-local/runtime-litert` consumers.

From the playground model list choose **Parakeet TDT · MediaPipe Audio Transcriber**. `ParakeetAsrPanel` can transcribe a browser-decoded audio file, or 16-kHz mono PCM from an AudioWorklet microphone stream. `finishStream()` runs after a microphone flush/queue drain; interim/final segments are accumulated by time interval. Existing Nemotron ASR stays separate and unchanged.

## Asset provenance
- Model: `litert-community/parakeet-tdt-0.6b-v3`, commit `1f78e5eb6175b1b2cce35f16bcf9e2027b83e8bd`.
- Artifact: `parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm`; size `614588752` bytes; publisher's LFS SHA-256 `66c5a74cf6c2c1d2b990901f6cece5bb53e3214dbceafa0989900a6f0eeb3504`.
- Runtime: `@mediapipe/tasks-audio@1.2.2`; JS package installed through pnpm, WASM requested from a version-specific jsDelivr URL after Load.
- Model weights licensed CC BY 4.0; MediaPipe Tasks Apache 2.0.

## Boundary and verification
Model and WASM are **not preloaded**. The user's explicit Load action fetches the model if a local `.litertlm` File is not supplied. After assets are loaded, transcription uses local WASM and does not call a cloud ASR API. This version does not maintain its own offline asset cache or verify the browser-fetched Hugging Face weight bytes against the pinned SHA; revision pinning alone is **not** a checksum. A manually chosen `.litertlm` file is checked for extension/nonempty input, not cryptographic identity.

Tests were RED for missing substrate, then green. On this checkout `pnpm --filter playground typecheck`, `pnpm --filter playground test` (206 tests), `pnpm --filter playground build` and `pnpm verify` passed. These verify package, UI compilation, parsing/accumulator contracts, and build/boundary/pack checks, **not actual speech inference**.

## Remaining acceptance
On desktop, open the actual playground in Chromium; with explicit space/bandwidth permission, load the exact model, transcribe a short voice WAV, then test microphone start/stop on a secure origin. Record transcript correctness, interim-replacement behavior, peak RAM, real-time latency, caching/network requests after load, and whether the engine can repeatedly start streams. No model was downloaded or inference exercised in this pass. Keep benchmarks separate from build-level success.
