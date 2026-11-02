# LFM3-Omni-2B LiteRT — Lil Local browser candidate (2026-11-01)

## Provenance and limitations

- Hub: `litert-community/LFM3-Omni-2B-LiteRT` (public metadata and its `litertlm_manifest.json` / `runner/contract.json` inspected, no weights downloaded).
- Published main bundle: `LFM3-Omni-2B_int8.litertlm`, 3,226,750,192 bytes, SHA-256 claimed by publisher `959617973a7ddcb3d887fa0d9891d79556a266490d5c432b02a18ffcac3124ed`. Neither bytes nor hash independently checked.
- The published bundle declares text, image, and audio input with text output; speech output requires additional graphs and its custom runner. Published min LiteRT-LM runtime version: 0.18.1. Neither that nor an Android report is proof of browser compatibility.
- On inspection `yuzu-mint` root filesystem had ~4.8 GiB free. No 3.23 GB download was attempted: model + caches/temp storage could exhaust the system drive.

## SDK compatibility discovery

`@litert-lm/core@0.19.0` TypeScript declarations expose `visionBackend`, `audioBackend`, and `ContentPart` image/audio inputs (video expressly not implemented in JS). This corrects the earlier assumption that the JS SDK only supports text. Actual multimodal model execution is **unverified**.

## Narrow worker improvement

Lil Local's LiteRT-LM worker previously rejected every non-text prompt. Added explicit opt-in `visionBackend: 'cpu' | 'gpu'` at model load and forwards image `Blob` content parts in messages only when vision was enabled. Text-only models retain their old default. Raw Float32 PCM audio remains rejected explicitly: SDK audio content uses encoded bytes/Blob, and an audio conversion path is not yet validated. MediaPipe's separate PCM handling remains intact.

Regression tests were run red/green for image-content forwarding and rejection when not opted into vision. The focused text-gen suite passed 93/93 and typecheck passed; full `pnpm verify` passed exit 0, including bundled consumer checks and build. These are tests with mocked native Engine; **no real LiteRT-LM image inference and no LFM3-Omni inference were run**.

## Next proof

Qualify with real model weights on a volume with sufficient free space, pin revision + verify SHA, run an actual Chromium session with real GPU/backend identification, and record text/image outputs, errors, peak memory, latency, disposal and re-load. Do not advertise LFM3 as browser runnable until that succeeds. Test audio input and speech as separate capabilities, not implications of the `.litertlm` file.
