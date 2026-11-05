# MediaPipe Text Language Detector — qualification receipt (2026-11-04)

## Provenance and reason to implement

Source: Google's MediaPipe Samples Web repository at **immutable commit**
`6887f6399745dc549dadd0022c31fc9ab01f26d6`;
inspected `src/tasks/language-detector.ts`,
`src/workers/language-detector.worker.ts` and
`tests/language-detector.spec.ts`. That upstream worker uses
`@mediapipe/tasks-text` and returns native language-code probabilities.

Lil Local previously had text embedding, retrieval, moderation,
ASR and TTS, but **no independent language-identification model**.
A small local model that scores the language of a supplied passage
is a reusable primitive for possible *future* transcript labeling,
TTS voice selection, and multilingual content handling. None of
those app-specific actions or policies are implemented here.

## Asset and backend

Official URL:
`https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite`

- Downloaded asset size: **315,294 bytes**, SHA-256
  **`7db4f23dfe1ad8966b050b419a865da451143fd43eb6b606a256aadeeb1e5417`**,
  calculated on the actual model file, not based on a guessed hash.
- The independent `@lil-local/text-mediapipe` package pins
  `@mediapipe/tasks-text@1.2.2` and matching WASM base
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@1.2.2/wasm`.
- Both byte length and complete SHA-256 are verified before native
  creation, including for caller-supplied local model bytes.
  No weights are committed to the repository.
- Default SDK `delegate:'GPU'` means **MediaPipe WebGL**, *not*
  native WebGPU. Explicit CPU option exists; initialization failure
  is surfaced instead of silently switching backends. The actual
  browser test used **SwiftShader-backed Chromium WebGL**. This is
  a functioning GPU-delegate API path, not evidence of physical
  GPU acceleration or any WebGPU performance claim.

## Public contract and source boundaries

The package exports
`createLanguageDetectorSubstrate()`,
`mapLanguageDetectorResult()`,
`validateLanguageText()`,
`verifyLanguageModelBytes()`,
`LANGUAGE_DETECTOR_MODEL` and the corresponding types.

The `BrowserSubstrateRunner` remains cold until explicit `load()`.
It accepts a nonblank Unicode text string (max 32,768 UTF-16 code
units) and returns
`{kind:'language-detection',languages:[{languageCode,probability},...]}`,
keeping **every native code/probability in native order** without
invented labels, translation, relabeling or score changes.
Invalid results and invalid probability values fail closed; empty
output remains empty. `dispose()` calls the native task's
`close()`, and pre-aborted loads never fetch weights.

The independently packed consumer can import, construct and run
the public contract without private workspace paths.
A separate opt-in playground text workbench offers preset samples,
text entry, explicit load/detect/unload and scored predictions.
It never requests camera permission; it does not auto-select voices
or modify any other Lil Local subsystem.

## Evidence

1. Seven contract tests were **RED before implementation** and then
   **GREEN**, including actual in-memory SHA-256 comparison, size
   rejection, native output shape, input/options validation, cold
   pre-aborted behavior, repeated inference via a mocked native
   lifecycle and exactly-once close. The mocked SDK case is *not*
   a substitute for the live-browser qualification.
2. The real `pnpm test:language:browser` launches disposable
   headless Chromium with production-built playground and the
   **actual versioned publisher model**. No model request occurred
   on page discovery or card opening. Explicit model load downloaded
   the model and the pinned Tasks Text WASM, successfully initialized
   the native **WebGL delegate**, and exposed real predictions:
   - English sample: top code **`en`**, displayed **100.0%**.
   - Spanish sample: top code **`es`**, displayed **100.0%**.
   - Japanese sample: top code **`ja`**, displayed **99.9%**.
   Three predictions reused the *same loaded native model*;
   explicit unload worked; no browser exceptions or unexpected
   console errors were observed. The browser profile was discarded.
   These are **three examples, not an accuracy benchmark**.
3. Final authoritative **`pnpm verify` exited 0** after the complete implementation, including all **7 language-ID unit tests**, **202 playground tests**, **18 boundary tests**, **73 runtime-qualification tests**, TypeScript checks, conversion checks and production builds. The independent pack/install/typecheck/test/build gate passed for all **24 packages**, including `@lil-local/text-mediapipe` imported from the packed public entrypoint. Existing Vite chunk-size/ineffective dynamic-import notices remain nonfatal.
4. The final checked-in **`pnpm test:language:browser` exited 0** with actual model/WASM downloads only after explicit activation, top native language codes `en`, `es`, `ja` for the corresponding full sentences, native model reuse, unload and zero browser exceptions.

## Remaining uncertainty

- The model classifies **one whole input**, not code-switching
  spans. Short or mixed-language inputs can be uncertain; results
  should never be treated as guaranteed language identification.
- Model quality on rarer languages, dialects, transliteration,
  misspellings, unusual punctuation or short strings was not
  assessed; no benchmark or latency comparison was performed.
- No Android device or physical GPU acceleration has been
  qualified. WebGL is not WebGPU, and this SDK does not expose
  a native WebGPU delegate for language detection.
- MediaPipe may emit telemetry. While the adapter never
  intentionally uploads input text, fully offline/no-telemetry
  behavior depends on the SDK and locally hosted WASM/model.
- No files in other repositories were changed; no consumer apps
  were integrated; no pushes, merges or releases occurred.
