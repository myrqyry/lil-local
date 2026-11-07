# @lil-local/audio8-tts — pinned candidate, not inference-ready

Audio8 Realistic Air 0.6B LiteRT **asset/provenance integration only**.

This package exports `createAudio8Manifest()` and three named profiles. It does **not** export an inference pipeline or imply that LiteRT.js can run the publisher's LiteRT 2.2 Android graphs. Do not put a working Generate control in a consumer on the strength of this manifest.

Upstream: https://huggingface.co/litert-community/Audio8-TTS-Realistic-0.6b
Revision: `752cd650bf5f649f2c0139ea02ac5627e9e60a5a`
The manifest pins published sizes and SHA-256 digests from `SHA256SUMS`; these were checked as metadata, **not by downloading and hashing the multi-GB weights locally**.

## Download profiles

- `synthesis` (default): tokenizer, slow/fast AR, 128-frame decoder, English/Japanese voice codes and transcript metadata.
- `extendedDecode`: synthesis plus the 192-frame decoder.
- `voiceRegistration`: synthesis plus 10-second codec encoder for registration. Registration still requires a correct reference transcript and a working host inference loop.

No profile silently downloads optional multi-hundred-MB graphs. The verification remains `unverified` across all profiles. `memory.residentBytes` is an Android-derived planning estimate, **not** measured browser RAM.

## Qualification required before enabling a browser pipeline

1. Verify LiteRT.js operator/signature support (slow AR's `prefill_256`, `prefill_64`, `decode`, plus dynamic KV state); determine actual available backend per graph.
2. Port and test upstream prompt formatting, tokenizer, repetition-aware sampling, slow+fast DualAR scheduling, EOS and frame cap.
3. Port the neural codec framing and decoding with a real reference vector. Compare CPU/WASM output first, then independently test WebGPU parity before advertising acceleration.
4. Run an actual browser load + English/Japanese synthesis with waveform, rate, duration, cancellation, disposal, and observable memory/telemetry receipts. Test voice registration separately.
5. Add explicit opt-in playground download and synthesis controls **only after** the actual browser path works. Preserve a nonworking candidate as unqualified rather than a misleading app feature.

The GGUF conversion from Hansda is **not** part of this package: it requires audio.cpp, not the current LiteRT runtime.
