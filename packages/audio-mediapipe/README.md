# @lil-local/audio-mediapipe

Reusable **opt-in** browser inference with the real
`@mediapipe/tasks-audio@1.2.2` SDK and matching version-pinned WASM.
Lil Local owns reusable inference and lifecycle contracts. Downstream
applications own microphones, audio capture/worklets, playback, episode logic,
OBS actions and user consent. Importing this package does not fetch a model.

## Current task families

| Task | Public API | Evidence |
| --- | --- | --- |
| Sound event classifier (YAMNet, 521 classes) | `createYamNetSubstrate()` | **Real Chromium CPU/WASM inference** on a generated 440 Hz WAV, official 4,126,810-byte model SHA-256 verified |
| Parakeet TDT 0.6B speech transcription | `createParakeetSubstrate()` (one-shot); `createParakeetTranscriber()` (streaming SDK engine) | Unit/packed tests only; **real 615 MB model and speech inference unverified** |
| MediaPipe text-to-speech | `createMediaPipeTtsSubstrate({model})`; `createMediaPipeTtsEngine({model})` | Installed SDK declaration/pack tests only; **no TTS model has been qualified** |

This does not replace `@lil-local/kokoro`, `@lil-local/phonon`,
`@lil-local/qwen3-tts` or the ONNX Nemotron browser ASR path. MediaPipe
TTS provides an alternative **LiteRT-LM bundle** engine with `listVoices()`,
`synthesize()` and `synthesizeStream()`. No automatic "best model" selection
is done; callers supply the `.litertlm` model by URL, Blob or Uint8Array,
and can choose `delegate: 'CPU' | 'GPU'` explicitly. Failed GPU setup does
not fall back to CPU silently.

### External consumer usage

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createYamNetSubstrate, createParakeetTranscriber,
  createMediaPipeTtsEngine } from '@lil-local/audio-mediapipe'

const classifier = new BrowserSubstrateRunner(createYamNetSubstrate({
  // model: officialYamNetBlob, // optional verified local bytes
  // wasmBase: '/mediapipe/wasm', // must be matching Tasks Audio 1.2.2 files
}))
await classifier.load()
const result = await classifier.run({
  samples: audioMonoFloat32,
  sampleRate: 16000,
})
console.log(result.windows[0]?.heads[0]?.categories)
await classifier.dispose()

// Real stream API, managed by caller, with microphone PCM + flush:
const asr = await createParakeetTranscriber(optionalLocalBundleFile)
await asr.prewarm()
await asr.transcribeStream(audioMonoFloat32)
const ending = await asr.finishStream()
console.log(ending.text)
asr.close()

// Explicit .litertlm model required; no surprise model download:
const tts = await createMediaPipeTtsEngine({
  model: officialCompatibleTtsBundle,
  defaultVoiceId: 'af_heart',
})
const voices = await tts.listVoices()
const speech = await tts.synthesize('Hello there')
tts.close()
```

The direct SDK engine factories are intentional for consumers that need
native streaming methods. The `BrowserSubstrateRunner` wrappers are for
simple one-shot inference with the standard Lil Local session lifecycle.

### Data, asset, and safety boundaries

- **YAMNet:** published official float32 model v1 pinned to
  `https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite`.
  Actual SHA-256
  `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`.
  The browser verifies **both** length and digest before creating the SDK
  task. Passing a local Blob/Uint8Array must satisfy the same verification.
- **Parakeet:** model revision
  `1f78e5eb6175b1b2cce35f16bcf9e2027b83e8bd`, about 615 MB.
  Revision pinning is *not* a hash check. The model URL is currently handed
  to the SDK and its contents are unverified on the browser path; local
  bundles are checked for nonempty input / `.litertlm` file extension, not
  full SHA. Do not claim parity, quality or speed from tests alone.
- **TTS:** only supplies a caller-selected model; the model is *not* assumed
  to be compatible. CPU/GPU and voice must match the bundle.
- No camera or microphone is opened by any package function; consumer
  controls capture/permissions and storage. Loading defaults to publisher
  HTTPS model/WASM endpoints; use model bytes and a matching self-hosted
  `wasmBase` for offline initialization. Avoid claiming the entire SDK is
  telemetry-free without a separate audit.
- MediaPipe's native tasks may block the browser main thread during inference.
  A consuming app requiring continuous interaction should run workloads in a
  real Web Worker; this package does not claim its caller is automatically
  worker-isolated. Audio input is normalized mono Float32.

### Verification

```bash
pnpm --filter @lil-local/audio-mediapipe test
pnpm --filter @lil-local/audio-mediapipe typecheck
pnpm test:compatibility
pnpm test:audio:browser
pnpm verify
```

The browser smoke uses a disposable Chromium profile, generated synthetic
440 Hz WAV, version-pinned CDN WASM and official Google YAMNet weights. It
requires system Chromium and Python `websocket-client`, opens **no mic**,
checks real classification time windows and unload behavior, then deletes its
own profile. Full receipt:
`docs/verification/2026-11-04-mediapipe-audio-substrate.md`.
