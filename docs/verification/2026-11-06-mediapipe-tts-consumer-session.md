# MediaPipe TTS streaming consumer session — Lil Local (2026-11-06)

## Purpose and scope
Mat wants **Lil Local as the reusable local inference substrate for podQast and Voxa**. Preserve each product's audio queue, mixer, voice-selection UX, narration timing, cancellation policy and microphone/OBS boundaries in its own app. Do not introduce another cloud-backed or app-specific synthesizer by accident.

This checkpoint adds an **opt-in, playback-neutral public session** to the existing `@lil-local/audio-mediapipe` package. It consumes the already-exported `createMediaPipeTtsEngine()` MediaPipe Tasks Audio 1.2.2 engine, **not** a fork of MediaPipe, an automatic new model loader, or another copy of the speech runtime.

## Public usage

```ts
import {
  createMediaPipeTtsEngine,
  createMediaPipeTtsStreamSession,
} from '@lil-local/audio-mediapipe'

const engine = await createMediaPipeTtsEngine({
  model: verifiedLocalBundle,  // explicit user-selected .litertlm bytes/Blob/URL
  wasmBase: '/mediapipe/tasks-audio/wasm', // matching self-hosted 1.2.2 WASM
  delegate: 'CPU',
})
const session = createMediaPipeTtsStreamSession(engine) // owns engine on close()
try {
  const voices = await session.listVoices()
  const controller = new AbortController()
  for await (const chunk of session.stream('Hello.', {
    voiceId: voices[0]?.voiceId,
    signal: controller.signal,
  })) {
    // Mono Float32 PCM, source sampleRateHz, chunkIndex, isFinal,
    // optional SDK wordTimestamps. Consumer owns resampling and playback.
    queueSpeechPcm(chunk) // example: product-owned playback implementation
  }
  // controller.abort() interrupts inference delivery; session.cancel() also works.
} finally {
  session.close()
}
```

## Verified contract / intentionally separate responsibilities
- At most one stream active per session; concurrent invocation rejects rather than mixing unrelated chunks.
- `cancel()` and an optional `AbortSignal` invalidate the current result even if native PCM resolves late. The adapter asks native `TextToSpeechStream.cancel()` / `TextToSpeech.cancel()` to stop compute; **native preemption is not guaranteed**.
- Explicit early iterator abandonment cancels the running native stream; `close()` cancels then releases native engine ownership once. Closed sessions refuse new work.
- No aliasing borrowed WASM PCM buffers: each chunk is normalized/finite-validated and copied to a caller-owned `Float32Array`. Preserve native sample rate, chunk index, final marker and optional word alignments.
- No model loads on import; models/voices are explicitly selected. No fallback to cloud, different model, or different delegate. No implicit audio output device or Web Worker. Callers must isolate generation from UI using their own worker when necessary.
- **Only complete text utterances** here. Raw SDK's `synthesizeStream('', ..., false)`, `pushText()`, `finishInput()` remain accessible via `createMediaPipeTtsEngine()`; a safe shared incremental-delta controller is future work, not claimed here.
- Unit tests use a fake native engine; they establish state transitions and boundary validation, not the sound of synthesized speech or native cancellation behavior.

## Downstream compatibility and unresolved work
- `podqast` at local `main` still pins older `@litert-playground/*` from `github:myrqyry/lil-tart#097b27a...`; its TTS worker produces request-correlated 24 kHz PCM16 and has an existing coordinator/watchdog. **Do not mix historic and renamed `@lil-local/*` packages in the same app.** A separate pinned single-revision migration, PCM16 conversion, cancellation and backpressure tests are necessary before wiring this session into podQast.
- The Voxa source repository was not found in the usual `~/loam` checkout; only `~/.local/share/voxa-rvc` was observed. No Voxa code or live connection was modified.
- Phonon 31M's prior real Chromium generation and streaming pass is documented in `2026-11-06-mediapipe-phonon-tts-browser.md`. That earlier pass tested raw MediaPipe engine methods, **not this new session in a real browser**. Real active cancellation, mobile latency, human auditory quality and long-run stability are still unverified.
- The machine's root volume was 99% full during inspection. No new model weights or dependencies were downloaded as part of this work.

## Verification receipt

- Wrote new package tests before implementation. Their first run failed because the exported session module did not exist (RED). New early-iterator-exit test also failed against the first implementation; corrected it (RED -> GREEN).
- `pnpm --filter @lil-local/audio-mediapipe test`: **18/18 passed** across 3 files, including six new session lifecycle cases.
- `pnpm --filter @lil-local/audio-mediapipe typecheck`: **exit 0**.
- `pnpm verify`: **exit 0**. This includes workspace typechecking/tests, WebGPU kernel probes, architecture boundaries, independently packed external consumer tests/typecheck/build, conversion assets, runtime qualification and production builds. The external consumer now imports the new public factory/type and runs a small fake-engine session through the packed package.
- `git diff --check`: clear. No new model weight or dependency installed. No actual TTS speech generation through this *new wrapper* was executed, so real SDK streaming proof remains limited to the earlier raw-engine browser test.
