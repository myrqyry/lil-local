# @lil-local/text-mediapipe

An opt-in, lightweight **language-identification** substrate using official
MediaPipe Tasks Text 1.2.2. This package does not perform transcription,
translation, segmentation of mixed-language passages, or automatic TTS routing.

## Qualifying model and runtime

Google's versioned `LanguageDetector` model is **315,294 bytes**:

`https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite`

Full SHA-256: `7db4f23dfe1ad8966b050b419a865da451143fd43eb6b606a256aadeeb1e5417`

The package verifies both downloaded byte length and full digest *before*
creating the native task. An explicitly supplied local `Blob` or
`Uint8Array` must match the same official hash. Nothing is downloaded
by importing the package or constructing a runner.

It pins `@mediapipe/tasks-text@1.2.2` with matching versioned
WASM; `wasmBase` can point at a self-hosted copy of that exact SDK.
The default `delegate:'GPU'` means **MediaPipe WebGL, not WebGPU**.
An explicit `delegate:'CPU'` is available, with **no hidden fallback**.
The generic `BrowserSubstrate` reports WASM because the inference-core
backend type doesn't include WebGL. Neither physical-GPU throughput nor
mobile performance has been established.

## Public consumer example

```ts
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {createLanguageDetectorSubstrate} from '@lil-local/text-mediapipe'

const runner=new BrowserSubstrateRunner(createLanguageDetectorSubstrate({
  maxResults:3,
  scoreThreshold:0,
  delegate:'GPU', // SDK WebGL, not WebGPU
  // model:officialLocalBlob,
  // wasmBase:'/mediapipe/tasks-text-1.2.2/wasm',
}))
await runner.load()
const result=await runner.run(
  'Esta biblioteca es un lugar tranquilo para leer un libro.')
// {kind:'language-detection',languages:[{languageCode:'es',probability:...},...]}
await runner.dispose()
```

`result.languages` preserves the SDK's original `languageCode` and
`probability` array **without inventing, translating or reordering
candidates**. Empty lists remain empty; invalid codes/scores fail closed.
Inputs must be nonblank strings of at most 32,768 UTF-16 code units.
The model classifies the *entire* passage. A mixture of English and Spanish
is **not** segmented by sentence, and very short strings may be
ambiguous; downstream consumers must choose any confidence thresholds,
confirmation or routing policy.

The shared package owns only native initialization, scoring, integrity
checks and disposal. The optional playground provides text input, preset
samples, explicit model loading and unload, and a native-score display.
No other app is modified or silently configured.

## Verification

```bash
pnpm --filter @lil-local/text-mediapipe test
pnpm --filter @lil-local/text-mediapipe typecheck
pnpm test:language:browser
pnpm test:compatibility
pnpm verify
```

The actual isolated-browser result, source commit, hashes and untested
boundaries are recorded in
[the verification receipt](../../docs/verification/2026-11-04-mediapipe-language-detection.md).

**Privacy caveat:** The adapter does not intentionally transmit input
text, but MediaPipe SDK telemetry or networking is not asserted absent
without auditing the SDK. Initial model/WASM download uses the network
unless the caller supplies verified local weights and self-hosted WASM.
