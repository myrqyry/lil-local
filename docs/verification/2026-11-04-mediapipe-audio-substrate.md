# Reusable MediaPipe Audio substrate — 2026-11-04

## Purpose and architecture

After bringing MediaPipe Tasks Vision into `@lil-local/vision-mediapipe`,
the corresponding Audio task catalog was reviewed. The prior Parakeet ASR
implementation was trapped in `apps/playground/src/substrates/parakeetMediaPipe.ts`.
Lil Local is the **shared inference substrate**; downstream projects should not
have to vendor a playground module or create new SDK wrappers. We preserved
existing Parakeet file/mic UI while moving its loader and segment reconciliation
into `@lil-local/audio-mediapipe`, then added the genuinely distinct official
YAMNet sound-event classifier. MediaPipe TTS was exposed as an opt-in engine,
not made a new default because `kokoro`, `phonon` and `qwen3-tts` already exist.

The public package exports:

1. `createYamNetSubstrate` with real
   `AudioClassifier.createFromOptions`, official model length+SHA verification,
   CPU/WASM, one-shot normalized PCM + sample-rate input, and time-window/head
   aligned categories. Supports verified local Blob/Uint8Array models.
2. `createParakeetTranscriber` (native streaming
   `transcribeStream`/`finishStream`/`prewarm` with caller-owned close),
   `combineParakeetSegments`, plus
   `createParakeetSubstrate` for one-shot `BrowserSubstrateRunner` users.
   Preserves the existing pinned publisher model revision and opt-in loading.
3. `createMediaPipeTtsEngine` (native `listVoices`/`synthesizeStream`)
   and `createMediaPipeTtsSubstrate` (one-shot) with explicitly supplied
   `.litertlm` model, requested CPU/GPU delegate, and validated audio result.

Native SDK startup/stop and model lifecycle stay in the package.
Capture permissions, web audio worklets, file selection, decoding, playback,
OBS actions and product policy stay in consuming applications. Calls to
`BrowserSubstrateRunner.load()` are explicit. No model weights are imported
or downloaded during discovery. An independent packed consumer now imports
all three actual entrypoints, and the playground uses the package directly.

## Asset identity

- MediaPipe Tasks Audio `@mediapipe/tasks-audio@1.2.2` with pinned matching
  `@mediapipe/tasks-audio@1.2.2/wasm`. No dynamic `@latest`.
- Official YAMNet float32 v1:
  `https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite`.
  Exact downloaded bytes: **4,126,810**.
  Locally computed SHA-256:
  `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`.
  Actual browser fetch is checked against *both* size and digest before
  model creation. Wrong payload and wrong lengths are unit tested to fail.
- Parakeet model:
  `litert-community/parakeet-tdt-0.6b-v3`, pinned revision
  `1f78e5eb6175b1b2cce35f16bcf9e2027b83e8bd`;
  official LFS metadata reports 614,588,752 bytes and SHA
  `66c5a74cf6c2c1d2b990901f6cece5bb53e3214dbceafa0989900a6f0eeb3504`.
  These are provenance facts, **not a browser-fetched hash verification**.
  Current SDK URL path does not verify the downloaded 615 MB buffer;
  local model files are verified as nonempty with the correct extension,
  not by SHA.
- TTS requires the caller to provide a compatible model URL or bytes.
  No model was chosen automatically and no TTS checkpoint was fetched.

## Evidence

- Tests written RED-first for missing Audio package and a separate invalid
  TTS PCM edge case; green after implementation. **12/12** Audio package
  tests across SDK pinning, YAMNet byte integrity, per-window mapping,
  input validation, TTS output validation, Parakeet interim/final
  reconciliation, cold substrate creation, and pre-aborted load behavior.
- Audio package typecheck passes. Playground typecheck passes and its
  **202/202** remaining tests pass; the original 4 Parakeet tests now live
  under `packages/audio-mediapipe`.
- **Packed external consumer**: actual `pnpm pack` of 22 packages into
  a fresh consumer, independent installation, public TypeScript import
  and execution of all three cold model descriptors, typecheck and build.
  `pnpm test:compatibility` exit 0.
- **Real Chromium WASM + model inference**: the production preview was
  opened in an isolated Chromium profile. The YAMNet card did *not*
  download model bytes before explicit Start; clicking Load requested
  official model and version-pinned MediaPipe Audio WASM. The browser
  verified SHA-256 and initialized native model execution. A generated
  2-second, 440 Hz, 16 kHz mono PCM WAV was uploaded (no microphone).
  Three real classification time windows were returned:
  - Window 1, 0.00 s: `Sine wave` probability **0.97**
  - Window 2, 0.97 s: `Sine wave` probability **0.97**
  - Window 3, 1.95 s: `Busy signal` probability **0.41**
  These are observed single-run model results, **not accuracy benchmarks**.
  Unload disposed the native classifier; no browser runtime exceptions
  remained after excluding the known informational XNNPACK log.
  Script: `scripts/mediapipe-audio-browser-smoke.py`,
  command: `pnpm test:audio:browser`.
- Whole repository `pnpm verify`: **exit 0** on completed code,
  including package typechecks/tests, 18 boundary tests, independent packed
  consumer, conversion-asset validation, 73 runtime-qualification tests
  and production builds. Existing Vite large-chunk and upstream
  ineffective dynamic-import notices remain advisory.

## Unverified or intentionally excluded

- Real 615 MB Parakeet inference, real speech transcript correctness,
  microphone streaming latency or finalization in the browser, Android
  performance, and actual TTS bundle initialization/synthesis are *not*
  established by passing TypeScript or SDK API tests. They need separate
  real-media/model qualification, not placeholder/mocked outputs.
- YAMNet recognizes only its 521 trained event classes. Audio classification
  is not a transcription, diarization, general audio-understanding, or speech
  synthesis engine.
- TTS model compatibility must be checked for the actual LiteRT-LM
  checkpoint/voice/delegate; `createMediaPipeTtsEngine` is intentionally
  low-level and exports native voice enumeration and streaming methods.
- MediaPipe CPU inference can block the main thread. Continuous capture apps
  should move the package calls to their own real worker and validate
  concurrency; no worker migration was claimed here.
- No consumers in ViiRM, Stagehand, podQast, or Android Meristem have been
  changed. The package will need a reachable pinned Lil Local Git SHA for
  those repositories to consume it. No push, release, or dependency-pin
  publication is performed during this work.
