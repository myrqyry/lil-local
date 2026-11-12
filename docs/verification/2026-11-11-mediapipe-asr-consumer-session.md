# MediaPipe AudioTranscriber consumer session — 2026-11-11

## Goal and ownership

Make Lil Local's existing MediaPipe Parakeet ASR path reusable by apps such as
Veyax, without moving microphone/tab capture, video timeline, model choice, or
commentary decisions into the inference package.

This is a **stream-lifecycle and public-consumer qualification**, not a claim
that the 614,588,752-byte Parakeet model has transcribed real speech.

## Current implementation

- `@lil-local/audio-mediapipe` now exports
  `createMediaPipeAsrStreamSession(engine)` with
  `push(Float32Array)`, `finish()` and asynchronous `close()`.
- AudioWorklet buffers are copied when submitted, serialized before native
  `transcribeStream`, and drained before `finishStream`; transcript snapshots
  copy segments and merge interim/final text by time.
- Closing immediately invalidates late deliveries, but waits for native calls
  before disposing the supplied engine. This does not interrupt native WASM
  computation already in flight.
- Default session ownership closes the native engine; callers reusing an
  engine for file transcription can specify `closeEngineOnClose: false`.
- The existing `ParakeetAsrPanel` microphone path now consumes this public
  session rather than maintaining a parallel inference queue. Its existing file
  path and model-loading controls remain independent.
- Packed external-consumer fixtures import, typecheck and exercise the public
  API without downloading weights.

## Verification

- RED first: missing session module, then a failing non-owning-session
  regression. Both were resolved without skipping tests.
- Focused `@lil-local/audio-mediapipe` package: **23/23 tests pass**.
- Package and playground TypeScript checks: pass.
- Root `pnpm verify`: **exit 0**, including typechecking, package tests,
  WebGPU kernel heuristic tests, boundaries, packed external compatibility,
  conversion-asset validation, runtime qualification and production build.
  Existing bundler chunk-size warnings remain advisory.
- **Not verified:** Parakeet model execution, microphone permissions/device
  capture during this change, long recording stability, browser timing/latency,
  or Veyax consuming the new API.

## Downstream continuation and constraints

Veyax's `apps/web/package.json` currently pins all Lil Local packages to
the historical commit `cebfe18b`; that commit lacks the current MediaPipe
packages. Its YouTube share source still requests `audio: false`. To make
Veyax hear a video, first qualify a real ASR model and the audio-capture
permission flow, then coherently repin **all** of its Lil Local packages to a
single verified revision and wire the shared inference API behind the existing
browser inference coordinator. Never install a duplicate native MediaPipe
transcriber inside Veyax or silently switch to a cloud provider.

No large model downloads, Veyax edits, deployment, or Git push were performed.
At this checkpoint the Loam filesystem reports approximately 2.7 GB free;
do not casually download 615 MB+ model artifacts.
