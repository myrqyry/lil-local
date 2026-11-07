<!-- meristem-template:v1 -->
# Active Work

## 2026-11-06 — Phonon 31M MediaPipe TTS real browser qualification
- On branch `aster/mediapipe-tts-qualification` from the verified Audio8 candidate `866edb8`, added an **opt-in browser smoke** that imports the existing Lil Local `createMediaPipeTtsEngine` public implementation and serves the external, already-cached `phonon_int8.litertlm` via a loopback-only temporary Vite endpoint. No model download, production UI, other app integration or hidden fallback.
- Verified real 87,205,888-byte model SHA-256 `b33a9a0d65acec36762db38a2bff6d899f4aab34b196bb56d89fab5665eabebe` matched Hugging Face LFS metadata. Chromium MediaPipe 1.2.2 CPU/WASM actually synthesized 109,440 PCM samples (4.56 seconds at 24 kHz), peak 0.5149, RMS 0.07156, and 10 word timestamps; 3.26 sec unary inference on the sampled run. Streaming yielded 19 ordered chunks/113,280 samples with final marker. Same engine switched from freya to claire and returned 72,960 non-silent samples. Browser page exceptions absent.
- Current publisher docs call Phonon English-only; actual bundle `listVoices()` surfaced locales en-US, fr-FR, de-DE, es-ES, pt-BR. French-labeled speech generation passed, but **intelligibility and language accuracy are not established**. No human listening, active cancellation, worker isolation, WebGPU, or Android performance verification.
- Repo-wide `pnpm verify` **exit 0** with the fixture/browser harness (prior to documentation-only changes). Exact reproduction/limits: `docs/verification/2026-11-06-mediapipe-phonon-tts-browser.md`. No push/merge/deploy/release.

## 2026-11-06 — Audio8 Realistic Air LiteRT candidate asset contract
- Added `@lil-local/audio8-tts` as an **asset/provenance-only** package on `aster/audio8-litert-candidate`, from LiteRT-community revision `752cd650bf5f649f2c0139ea02ac5627e9e60a5a`. The publisher's Android LiteRT 2.2.0 route is *not* evidence of LiteRT.js browser inference.
- Asset plans: synthesis (tokenizer, slow/fast AR, T128 codec, two bundled voice entries); extendedDecode (adds T192 codec); voiceRegistration (adds 10-second encoder only on explicit selection). Profiles pin byte sizes and SHA-256 for ten assets. Independently compared all ten digest strings against the pinned `SHA256SUMS` on the publisher; no model weight bytes were downloaded or hashed locally.
- RED→GREEN: missing manifest failed import; 3 focused tests and package typecheck pass. Full `pnpm verify` **exit 0** after introducing the package and lockfile. These checks prove package hygiene and manifest behavior, not audio synthesis.
- Still needed: LiteRT.js graph/signature/operator qualification, prompt/tokenizer port, DualAR host loop and cache lifecycle, codec parity, actual browser PCM output and memory/backends, plus opt-in UI only if successful. GGUF/audio.cpp remains a distinct unimplemented substrate.
- Receipt: `docs/verification/2026-11-06-audio8-candidate.md`. No push, merge, release, model download, or downstream app integration.

## 2026-11-04 — Local MediaPipe language identification as an independent text substrate
- Continued Google's official MediaPipe Web Samples investigation at pinned upstream `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`, inspecting the Language Detector task, worker and browser test. Lil Local previously supported text embedding/retrieval, ASR and TTS but had **no distinct lightweight language-ID task**; it can later support transcript labeling or voice-selection decisions without embedding those policies into the shared inference engine.
- Added separate `@lil-local/text-mediapipe` package (not in Vision) with `createLanguageDetectorSubstrate()`, `mapLanguageDetectorResult()`, typed unchanged scores/codes, strict Unicode input and option validation, explicit lazy load/unload and full model-size+SHA-256 validation including user-provided official weights. Publisher model **315,294 bytes**, SHA-256 `7db4f23dfe1ad8966b050b419a865da451143fd43eb6b606a256aadeeb1e5417`. Pinned `@mediapipe/tasks-text@1.2.2`, same-version WASM; selected default SDK `GPU` means **MediaPipe WebGL not WebGPU**, optional explicit CPU, no hidden fallback. Real Chrome used SwiftShader, not a physical-GPU benchmark.
- Playground has opt-in **Language Detector · MediaPipe Text** text box, English/Spanish/Japanese samples, explicit load/detect/unload, native confidence outputs, no camera or transcript modification. **Real Chromium** with the actual publisher model reported `en` 100.0%, `es` 100.0%, `ja` 99.9% on the three samples; one model session was reused, browser errors 0, native unload worked. Model was never requested on import/card selection. Reproduce `pnpm test:language:browser` **exit 0**. These are examples, not multilingual accuracy benchmarks; input is a whole passage rather than per-sentence code switching.
- Seven new tests **RED→GREEN**; playground **202/202**; full **`pnpm verify` exit 0**, including 18 boundary cases, independently packed **24** packages with external consumer public API, 73 runtime qualification tests and production builds. Evidence receipt `docs/verification/2026-11-04-mediapipe-language-detection.md`. No other app/project integration or push/merge/release.
- Local branch `aster/mediapipe-language-detection` continues verified BlazeFace commit `e80946c`; preserve focused local Git checkpoint.


## 2026-11-04 — BlazeFace Short Range lightweight face ROI detector in shared Vision
- Continued official MediaPipe Web Samples at fixed `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`. Picked distinct lightweight face **boxes and six keypoints** rather than duplicating full 478-point face meshes. Upstream face detector test used a *dog* image, so real qualification uses pinned portrait and asserts an actual detection.
- Added `createFaceDetectorSubstrate()`, `mapFaceDetectorResult()`, `MEDIAPIPE_VISION_MODELS.blazeFace` within the existing `@lil-local/vision-mediapipe`. Official float16 publisher file **229,746 bytes**, SHA-256 `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`. Reuses full integrity verification and native IMAGE/VIDEO lifecycle, preserves native pixel boxes/score/normalized keypoints, explicit native unload. GPU delegate selected by default and in playground = **MediaPipe WebGL, not WebGPU**, explicit CPU alternative, no silent fallback. Headless Chrome runs SwiftShader; no physical GPU performance claim.
- Existing Vision playground now offers an **eighth** frame task, BlazeFace still-image boxes/keypoint overlay without webcam permission, plus synthetic-camera video. **Real Chromium** on upstream `public/face_model.png` SHA-256 `75171e877e92b7a126cca2e7a388fc430225e07e9cd2e9e801eaa67ea6d7f4d9`: **one face, 0.94 confidence, 215×215 pixel box, six keypoints**. The more distant human `pose_model.png` produced **zero detections**, a real short-range limitation explicitly retained. All seven other video paths and image cases passed in browser smoke; no user webcam opened.
- Six focused tests RED→GREEN; Vision **49/49**, playground **202/202**. Packed public external consumer extended to BlazeFace; full **`pnpm verify` exit 0**, including 18 package boundary tests, independently packed 23-package external consumer, 73 runtime qualification tests, conversion checks and production builds. Receipt `docs/verification/2026-11-04-mediapipe-blazeface-detector.md`.
- Branch `aster/mediapipe-blazeface-detector` based on checked-in MagicTouch checkpoint `9a9d5f5`; preserve as focused local commit, no push/merge/release or unrelated app integrations.


## 2026-11-04 — MagicTouch V2 interactive segmentation on a cached image
- Continued Mat's official [MediaPipe Web Samples](https://github.com/google-ai-edge/mediapipe-samples-web) exploration at pinned upstream `6887f6399745dc549dadd0022c31fc9ab01f26d6`. Compared upstream `InteractiveSegmenter.setImage()` + `segment(strokes)` with Lil Local's existing signed point-prompted SAM 2.1 (`packages/image-segmentation/src/runtime.ts`). SAM currently exposes one-shot prompt calls without an explicit reusable encoded-image session. MagicTouch differs by **encoding once then applying repeated native stroke refinements**; no claim it beats SAM in accuracy or speed, and existing SAM remains unchanged.
- Extended existing `@lil-local/vision-mediapipe` with `createInteractiveSegmenterSubstrate()`, typed `set-image`/`segment` requests and `copyInteractiveMask`. Actual official MagicTouch V2 int8 asset: **30,525,312 bytes**, SHA-256 `38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03` verified locally. Reuses exact-byte/digest loader and pinned Tasks Vision 1.2.2 CPU/WASM. Strict normalized cumulative include/exclude/lasso strokes; soft float masks copied before native mask close; explicit state and unload. No implicit model load, webcam, persistence, or SDK replacement.
- Separate opt-in playground model lab **MagicTouch V2 · Interactive Segmentation** for local image uploads and in-image include/exclude/lasso strokes, Undo/Clear and native Unload. Kept the seven continuous Vision camera tasks separate. External independent packed consumer constructs and tests the new public API.
- Focused six tests RED→GREEN with mocked native lifecycle: verified **one set-image call reused by multiple segment calls**, input bounds and mask ownership/close. Shared Vision **43/43**; playground **202/202**. Full `pnpm verify` **exit 0**, including boundaries, 23 independent packed packages, 73 runtime qualification cases and production builds. The browser smoke is `pnpm test:interactive:browser`, source/limitations in `docs/verification/2026-11-04-mediapipe-interactive-segmentation.md`.
- Real Chrome on upstream SHA-verified dog photo loaded publisher model and native SDK, encoded once and returned **640×640** real soft mask. Positive stroke selected **42.4%** of pixels above 0.5; negative stroke over the dog shrank selected fraction to **8.0%**. Final `pnpm test:interactive:browser` **exit 0**, including assertion of −34.4 percentage-point coverage reduction; Undo, clear and unload exercised. Lasso only unit-tested, not real-model qualified. **Performance limitation:** encoding took ~3.4–4.6s; individual strokes ~5.6–8.2s on desktop headless CPU/WASM+SwiftShader. Explicitly **not instant** or a SAM benchmark, no mobile qualification. SDK logs include harmless OpenGL/feedback-manager messages surfaced through console.error; test filters only known exact strings, still fails other runtime errors.
- Branch `aster/mediapipe-magic-touch` based on verified DeepLab commit `b0996cc`. Local focused commit only; no pushing, merging, releases or cross-app integrations. Document the alternative as experimental given refinement latency.


## 2026-11-04 — DeepLab V3 semantic class masks in the shared Vision runtime
- Continued the official MediaPipe Web Samples investigation at pinned upstream `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`, inspecting image segmenter task, worker, native mask lifetime API and Playwright fixture. This is categorically different from existing SAM point-prompted foreground/instance masks: the returned information is an **ID for every semantic pixel**.
- Extended existing `@lil-local/vision-mediapipe` with `createSemanticSegmenterSubstrate()`, `copySemanticCategoryMask()` and verified official DeepLab V3 float32 model (exact **2,780,176 bytes**, SHA-256 `ff36e24d40547fe9e645e2f4e8745d1876d6e38b332d39a82f0bf0f5d1d561b3`). Shares full-hash model loader and Tasks Vision 1.2.2 CPU/WASM; copies native borrowed pixel data *before callback returns*; preserves model class IDs/labels, with no invented category or confidence and validated bounded mask dimensions. Explicit IMAGE/VIDEO and close, no implicit loads.
- Existing playground now offers a seventh opt-in Vision task with local still-image mask overlay, class-share legend and real SDK text labels; no camera permission for still images. **Real Chromium** on Google upstream SHA-verified dog photo produced **640×640 semantic mask: background class 0 59.2%; dog class 12 40.8%**. SVG/canvas class overlay drew 167,314 nontransparent pixels. The native VIDEO path and six earlier Vision tasks passed with fake webcam, camera cleanup and no unexpected browser errors. Reproduce `pnpm test:vision:browser`.
- Five new tests RED then GREEN; Vision **37/37**, playground **202/202**; packed downstream consumer now covers the semantic public API. Full `pnpm verify`: **exit 0** (37 Vision tests, 202 playground tests, 18 package boundary tests, 23 packed packages, 73 runtime-qualification tests and production builds). Source/test/limits receipt `docs/verification/2026-11-04-mediapipe-semantic-segmentation.md`.
- Branch `aster/mediapipe-semantic-segmentation` based on verified holistic `3862000`; preserve a focused local commit, do not push/publish or integrate into unrelated projects.


## 2026-11-04 — Holistic face/body/hands in one local MediaPipe session
- Continued working from Google's official MediaPipe Web Samples at pinned upstream `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`, reviewing the native Holistic task and SDK declarations. A standalone hand landmarker would overlap the gesture runner; Holistic is distinct because **one model session produces coordinated face, body and explicitly sided hand observations**.
- Extended the existing `@lil-local/vision-mediapipe` package with `createHolisticLandmarkerSubstrate()`, `mapHolisticLandmarkerResult()` and a new `holistic-landmarks` capability. Official 13,683,609-byte float16 checkpoint, SHA-256 `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8`. Reused existing full-byte+digest model integrity, CPU/WASM 1.2.2, IMAGE/VIDEO, explicit native cleanup; validated cross-channel geometry, probabilities, counts and confidence thresholds. Did **not** synthesize gesture classifications from raw landmarks.
- Existing playground gained a sixth choice, still-image Holistic overlay with native face blendshapes/pose/side-labeled hands. **Real Chromium** inference of SHA-verified upstream human pose image returned **478 face landmarks** with actual blendshapes, **33 image + 33 world pose points**, **21 image + 21 world points each for left and right hands**, **553 overlay dots**. Native VIDEO path also ran with a fake webcam, cleanup succeeded and previous five tasks still passed. Reproduce `pnpm test:vision:browser`.
- Seven tests RED then GREEN; Vision **32/32**, playground **202/202**, independent packed consumer **23 packages**, full `pnpm verify` **exit 0**. Receipt `docs/verification/2026-11-04-mediapipe-holistic-landmarker.md`. Does not establish continuous accuracy, user-camera semantics, animation retargeting, or mobile performance.
- Branch `aster/mediapipe-holistic-substrate` based on qualified image-classifier checkpoint `01b2dbb`; local focused commit, no push or other repo changes.


## 2026-11-04 — MediaPipe EfficientNet-Lite0 whole-image classifier in shared Vision
- Continued directly from Google's [official MediaPipe Web Samples](https://google-ai-edge.github.io/mediapipe-samples-web/): upstream `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`, image-classifier task, worker and dog-image Playwright test. This is a distinct task from EfficientDet's localized object boxes and from Lil Local's existing video classifier; do not conflate them.
- Added `createImageClassifierSubstrate()`, `mapImageClassifierResult()`, and official `MEDIAPIPE_VISION_MODELS.classifier` within **existing** `@lil-local/vision-mediapipe` (no new runtime). EfficientNet-Lite0 float32 official model: **18,582,189 bytes**, SHA-256 `6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0`. Shared byte/digest verification, lazy model loading, CPU/WASM 1.2.2, IMAGE/VIDEO modes, native cleanup; output retains per-head indices/names and category scores without fabricated boxes.
- Playground now includes a fifth local Vision task; still image requires no camera permission. **Real Chromium** on the immutable upstream dog photo returned `golden retriever` score **0.85**, then VIDEO ran on the fake webcam. Face/gesture/pose/object tasks still passed, with verified real pose and dog-box examples. The independent packed consumer also imports/constructs the new public runner.
- Six tests RED then GREEN; Vision **25/25**, playground **202/202**; authoritative `pnpm verify` **exit 0** (package tests/typecheck, boundaries, 23 packed packages, qualification, conversion, builds). Full browser command `pnpm test:vision:browser`; exact limitation/source receipt `docs/verification/2026-11-04-mediapipe-image-classification.md`. No general accuracy, mobile or physical-camera claims; SDK telemetry/offline boundaries unchanged.
- Local branch `aster/mediapipe-image-classifier` continues verified object checkpoint `4f8fbdc`; changes confined to Lil Local, no push, merge, release or other-app integration.


## 2026-11-04 — MediaPipe EfficientDet-Lite0 object detection in shared Vision
- Continued the [Google MediaPipe web gallery](https://google-ai-edge.github.io/mediapipe-samples-web/) investigation **within Lil Local**, examining upstream object detector, worker and Playwright dog fixture at `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`. Existing Vision included face/gesture/pose but lacked general object detection; no changes to other apps were made.
- Added `createObjectDetectorSubstrate()` and native `mapObjectDetectorResult()` inside `@lil-local/vision-mediapipe`, reusing its cold-loaded MediaPipe Tasks Vision 1.2.2 CPU/WASM lifecycle and full SHA verification. Official EfficientDet-Lite0 model: **13,836,895 bytes**, SHA-256 `40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58`. Preserves pixel-space boxes, per-object scores/categories, keypoints, IMAGE and VIDEO paths; no implicit webcam.
- Existing playground Vision panel now has object detection and pixel-aligned labeled box overlay on opt-in local image; positive **actual Chromium inference** on the official pinned dog photo detected `dog` score **0.84**, box **512 × 556 px**. Same isolated fake-camera test exercised face, gesture, pose and object VIDEO methods, with all four official model checks, no real user camera and successful cleanup. Reproduce with `pnpm test:vision:browser`.
- Six object contract tests RED then GREEN; Vision **19/19**, playground **202/202**. Packed downstream fixture includes the new public API; initial broad gate found a **test-only assertion** still expecting three idle runners after adding the fourth; corrected to four, independently packed consumer **23/23** passed. Full `pnpm verify`: **exit 0** (typechecks, tests, boundaries, packed consumer, conversion assets, qualification and builds).
- Evidence and limitations in `docs/verification/2026-11-04-mediapipe-object-detection.md`; branch `aster/mediapipe-object-detection` inherits pose checkpoint `8b5db73`; keep work local and preserve a focused Git commit, do not push merely as a side effect.


## 2026-11-04 — MediaPipe Pose Landmarker added to reusable Vision package
- Continued from Google's MediaPipe Web Samples gallery, **inside Lil Local**. Inspected upstream pose task, worker and Playwright source at `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`. No changes to ViiRM, Stagehand or other consumers.
- Added a verified `pose_landmarker_lite.task` manifest (5,777,746 bytes, SHA-256 `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`), one true `createPoseLandmarkerSubstrate()` using existing MediaPipe Vision 1.2.2 CPU/WASM and `BrowserSubstrateRunner` lifecycle, and precise image/world landmarks with optional native visibility/presence. Masks intentionally disabled.
- The playground consumes the same shared package for **local opt-in still-image analysis without webcam permission**, including pose-skeleton overlay; original face/gesture camera modes remain. One pinned upstream 640x640 test photo yielded **one pose with 33 normalized image landmarks, 33 world landmarks and 33 visible overlay points**. Fake webcam ran all three real SDK video tasks. Still image was ephemeral and SHA-verified; no personal camera accessed. Repeat: `pnpm test:vision:browser`.
- Six focused pose tests RED then GREEN; Vision 13/13, playground 202/202; independent packed-consumer fixture includes the pose API. Authoritative `pnpm verify` and browser smoke: completed successfully. Receipt: `docs/verification/2026-11-04-mediapipe-pose-substrate.md`.
- Branch `aster/mediapipe-pose-substrate` continues from verified retrieval `ec0122b`. Keep these capabilities reusable and don't force downstream integrations; local commit checkpoint, no push.


## 2026-11-04 — Official MediaPipe Web Samples → shared retrieval substrate
- Mat supplied the [official web sample gallery](https://google-ai-edge.github.io/mediapipe-samples-web/), pointing to reusable Lil Local capabilities rather than arbitrary integrations. Examined upstream `google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6` in temporary isolation; the meaningful novel pattern was **one shared Universal Embedder model/provider for Semantic Retriever**, while other gallery tasks overlap existing Lil Local packages or need their own qualification.
- Added `@lil-local/retrieval-mediapipe` (MediaPipe Tasks Retrieval 1.2.2), a real cold-loadable `BrowserSubstrateRunner` that supports typed text/image/audio embeddings, text/image/audio indexing, scored semantic retrieval, IDs/deletion, strict output validation, cancellation, and native cleanup. Created a thin playground client with explicit File/URL model selection and index/search, plus public packed-consumer integration; existing ColBERT and embedding packages unchanged.
- Downloaded **official** EmbeddingGemma 2 text 270M model (164,626,432 bytes) to `~/.cache/lil-local/mediapipe-retrieval/` only, SHA `2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb`, revision `9be6e8b90982095dc05c2bd162e4b954ee4dbac7`. Chromium with WebGPU available loaded the exact local model, returned a real **768-dimensional** vector and searched a native indexed document: `doc-1` score **0.839**. Zero browser errors, unload completed. Real image/audio inference and GPU placement were not qualified.
- Seven focused tests RED→GREEN, playground typecheck and 202 tests, packed consumer compatibility across **23 packages**, and reproducible `scripts/mediapipe-retrieval-browser-smoke.py`. Source mapping, evidence, and explicit remaining work: `docs/verification/2026-11-04-mediapipe-samples-retrieval.md`. Final full `pnpm verify`: exit 0 (workspace tests/typechecks, boundaries, 23 packed consumers, conversion, qualification, build). Local branch `aster/mediapipe-retrieval-substrate`; no push or cross-project changes.


## 2026-11-04 — MediaPipe Audio becomes a reusable Lil Local substrate
- MediaPipe's Audio catalog exposed the mismatch: our Parakeet ASR loader/segment logic was still playground-private although Lil Local is the shared inference substrate. Moved it and its four tests into the new `@lil-local/audio-mediapipe` package, without changing the working file/AudioWorklet UI path; added opt-in `createParakeetSubstrate` for one-shot external consumers.
- Added a truly new capability, official YAMNet sound-event classification: 4,126,810 publisher bytes with SHA-256 `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf` checked on browser load, real WASM CPU classification of a synthetic 440 Hz audio file (three windows; top `Sine wave` 0.97), opt-in playground entry and unload. Exposed an opt-in MediaPipe LiteRT-LM TTS engine/substrate with caller-selected model and native streaming/voice APIs, *not* a replacement for existing Kokoro/Phonon/Qwen3-TTS.
- RED-first tests for missing package and malformed native TTS PCM; **12/12** shared package tests, typecheck, playground **202/202**, packed external consumer with **22** packages, real Chromium synthetic-WAV YAMNet inference, and final `pnpm verify` all passed. No Parakeet ~615 MB model, microphone stream or MediaPipe TTS model was actually run; these remain unqualified. Documentation and exact reproduction command `pnpm test:audio:browser`: `docs/verification/2026-11-04-mediapipe-audio-substrate.md`.
- Local branch `aster/mediapipe-audio-substrate` continues on the verified vision checkpoint; changes confined to Lil Local, no cross-repo dependency mutation, no push.


## 2026-11-04 — Reusable MediaPipe Vision package, not another app-local runtime
- Added `@lil-local/vision-mediapipe` as a genuine `BrowserSubstrateRunner`-compatible, portable package for CPU/WASM face landmarks + per-face blendshapes and hand landmarks + gesture scores. Both official model artifacts are size/SHA-256 verified; JS/WASM pinned to 1.2.2, with optional caller-owned verified model weights and matching self-hosted WASM. No auto-download or camera access on import/construction, no OBS actions or character expression policy in the substrate.
- The Lil Local playground now **consumes that package**, offering opt-in camera tracking for face or hands; explicit Start/Stop, bounded single-in-flight frame inference, real cleanup. The external consumer fixture independently packs/installs and exercises its public contract, rather than importing workspace source.
- RED-first package tests now 7/7; playground 206/206, typecheck/build, external pack/typecheck/build/contract and initial authoritative `pnpm verify` all passed. Chromium production preview with synthetic webcam loaded and verified both Google models, initialized real MediaPipe WASM CPU inference for both tasks, ran actual frames, then stopped all camera tracks; no real face/gesture correctness or physical-device claims. An informational XNNPACK delegate log is not an exception.
- Receipt and boundaries: `docs/verification/2026-11-04-mediapipe-vision-substrate.md`; repeatable fake-webcam script: `pnpm test:vision:browser`. ViiRM and Stagehand remain separate consumer migrations: pin a reachable Lil Local SHA when ready, without committing absolute local paths. All work stays local on `aster/mediapipe-vision-substrate`.


## 2026-11-04 — Opt-in browser MediaPipe Parakeet speech transcription
- Added a separate Parakeet TDT 0.6B v3 browser ASR model entry/panel alongside Nemotron; implemented 16-kHz mono file and AudioWorklet microphone input with streaming `AudioTranscriber`, interim/final segment reconciliation, explicit model loading, optional local `.litertlm` file, and resource cleanup.
- Pinned `@mediapipe/tasks-audio@1.2.2` and publisher model revision; no automatic model download, no cloud transcription fallback. Browser weights/WASM are fetched on Load unless explicitly local; browser weight SHA and real speech inference remain unverified.
- Four new focused tests written RED-first, then green. Complete playground tests 206/206, TypeScript typecheck, production build and authoritative `pnpm verify` all passed. No real 615 MB model download or microphone/Chromium inference proof.
- Receipt and next acceptance: `docs/verification/2026-11-04-parakeet-mediapipe-asr.md`.

## 2026-11-03 — Consolidate surviving image-segmentation/moderation branches
- `aster/polish-moderation-backend-20261027` carried one still-missing correctness fix: the moderation substrate now reports only the backend actually requested from Transformers.js. Cherry-picked onto `master`; package tests 15/15 and typecheck pass.
- `aster/forma-image-segmentation` was not safe to merge wholesale because `master` now has a newer `@lil-local/image-segmentation` package for face parsing and SAM 2.1. Recovered only the still-missing portable U²-Net foreground-matte path (`mask.ts`, `pipeline.ts`, focused tests/manifest), wired it into the current public entrypoint, combined peer dependencies with the existing Transformers.js path, and added packed-consumer coverage. The matte path remains explicitly unqualified for real model execution/output quality.
- Evidence: image-segmentation tests 15/15, package typecheck, packed compatibility across all 20 supported packages, and full `pnpm verify` all pass. Existing Vite chunk-size/dynamic-import notices remain advisory.
- These recovered mechanisms now live on `master`; the old branches are historical evidence rather than required feature islands.

## 2026-11-02 — Promote verified Phonon experiment over later silent scaffold
- The preserved aster/phonon-live-onnx branch contains the real browser ONNX runtime verified on 2026-10-26. Current master later reintroduced the same package name as a structural scaffold whose own source says the graph ABI is unverified and whose run path emits one silent frame.
- Experiment unification therefore promotes only the verified package implementation and its focused browser/external-consumer receipts onto current master, while retaining today's workspace/package registry and regenerating the lockfile. This is not a wholesale merge of the older branch.
- Privacy Filter and Sharp ONNX remain separate follow-on experiment promotions after Phonon verification.


## 2026-11-02 — Browser speculative-decoding cutting

- **Objective:** Adapt the reusable parts of arXiv:2610.39761 (“GPU-Resident Speculative Decoding for In-Browser LLM Inference with WebGPU”) without claiming that the current LiteRT-LM or Transformers.js pipelines expose GPU-resident verification/KV control.
- **Implementation:** `@lil-local/text-gen` now exports exact suffix-continuation draft collection (default order/draft bounds 8), bounded model-scoped session token memory with caller-owned restore, a measured “smallest within 10% of fastest” submission-window selector, and an empirical speculation controller that chooses calibrated width from accepted tokens per millisecond, updates online measurements with a configurable exponential moving average, and fails closed to width 1 before calibration. The packed external consumer imports and executes these helpers. Personal session-token persistence is intentionally **not** routed through `browser-cache`; callers may persist snapshots in same-origin storage only when appropriate.
- **Evidence:** TDD captured RED states for the missing speculation module, missing session-memory restore, missing submission-window selector, and stale lifetime-average calibration that failed to respond to a recent width-2 slowdown. Final focused text-gen suite passes **105 tests** and package typecheck passes after the EMA/controller changes. Packed external-consumer compatibility passed for all supported packages, including `text-gen`. Final `pnpm verify` exited **0**, covering workspace typechecks/tests, WebGPU kernel-selection tests, package boundaries, compatibility, conversion assets, 73 runtime-qualification tests and production builds; existing Vite chunk-size/dynamic-import notices remained advisories. Because unrelated Nemotron playground files appeared concurrently, exact implementation commit `d94340d` was also verified from a fresh detached worktree with a frozen offline install; that independent `pnpm verify` also exited 0.
- **Boundary:** No `dot4I8Packed` small-batch kernels, GPU-side acceptance/tree walk, KV compaction, dual in-flight command windows, generated-token ring buffer, weighted GPU context-mixing trie, draft heads, exact greedy equivalence, or browser speedup is claimed. Those require a separately qualified custom WebGPU text runtime or new low-level upstream runtime hooks. Detailed receipt: `docs/verification/2026-11-02-gpu-resident-speculation-cutting.md`.


## 2026-11-01 — Verified Pix2Pix custom WebGPU package

- **Objective:** Promote the audited Pix2Pix U-Net 256 engine from a disposable verification workbench into the narrowest reusable Lil Local surface that the evidence supports, without bundling model weights or generalizing one checkpoint into universal Pix2Pix support.
- **Implementation:** Added `@lil-local/pix2pix-webgpu` with checked-in Rust/wgpu + WGSL source and its ~420 KiB wasm-bindgen runtime, caller-owned `AssetResolver` transport, strict folded-weight byte/hash verification, explicit BatchNorm-eval mode, fail-closed WebGPU-only loading, lifecycle ownership through `BrowserSubstrateRunner`, optional hidden canvas, final GPU NCHW readback, and RGBA8 image results. The playground exposes a separate **Pix2Pix · Local WebGPU** workspace rather than routing the custom runtime through the LiteRT model runner. Folded weights can now be imported/replaced once into the existing `litert-playground` OPFS store, detected without reading the 217 MB body, loaded after reload with no model picker, and explicitly removed. The verified host-side source copy lives at `$HOME/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-v1.bin`; browser inference uses its own origin-scoped OPFS copy. The model manifest remains intentionally `qualification: limited`.
- **Evidence:** The package's focused suite passes 8/8 tests. Browser-cache now has 11 passing tests including body-free presence probing, while Pix2Pix local storage has 4 passing tests including preservation of a good installed model when a replacement fails integrity. Playground typecheck and production build pass. A production Chromium smoke on NVIDIA/Turing imported the exact 217,677,836-byte folded model into OPFS, reloaded the app, observed an empty model file picker, rediscovered the installed model, loaded it only from OPFS, ran the full generator, rendered output, and recorded zero console/page errors. The actual OPFS entry was 217,677,836 bytes at the displayed deterministic cache location; the post-reload session receipt observed `load 517.7 ms`, `compile 391.5 ms`, `inference + readback 79.5 ms` on that run (not a benchmark). Earlier package-level byte parity and full floating-point parity evidence remains in `docs/verification/2026-11-01-pix2pix-webgpu-candidate.md`.
- **Boundaries:** Only the pinned folded official `edges2shoes` checkpoint is qualified; weights remain outside Git. OPFS is browser-origin scoped, so a different origin/port has a separate installed copy. The 253,233,196-byte manifest resident-memory figure is derived major static GPU allocation accounting, not process/driver peak VRAM. Mid-wgpu engine creation is not interruptible; late-created engines are freed after cancellation. No arbitrary checkpoint support, cross-GPU/browser qualification, or production latency claim. Final `pnpm verify` exited 0 after the persistent OPFS integration. No push/publication is implied.

## 2026-10-26 — Smallest working OpenAI Privacy Filter browser substrate

- **Intent:** Add an opt-in privacy detection and separate redaction capability usable from other Lil Local consumer apps. Provide device/precision choices to other users, but default to smallest **verified compatible** variant for the current browser; no server or automatic redaction.
- **Actual device result:** Chromium adapter supports WebGPU but **not** `shader-f16`, so `q4f16` (~809 MB) is incompatible with this browser capability; plain `q4` (~917 MB weights, plus ~28 MB tokenizer) loaded and inferred successfully. After whitespace-preserving alignment change, a second real Chromium run detected the synthetic name `Alice Smith` and address `alice@example.com` and generated `My name is [PRIVATE_PERSON] and my email address is [PRIVATE_EMAIL].` Test script exit 0. No claim of throughput or NVIDIA hardware GPU placement.
- **Implementation:** Experimental `@lil-local/privacy-filter` package with `createPrivacyFilterSubstrate`, explicit model load, mutable caller choice `dtype`/`device`, `resolvePrivacySpans`, and pure optional `redactPrivacySpans`. Repeated/normalized/missing/overlapping spans fail rather than mask an arbitrary occurrence; no originals modified.
- **Limits:** Transformers.js `simple` aggregation is **not** the model's reference constrained Viterbi decoder; model is a privacy aid, not anonymization assurance. Download needs Hub access until self-hosted; source pinned. Focused tests, independent packed consumer, browser runtime evidence documented in `docs/verification/2026-10-26-openai-privacy-filter-q4.md`. Work isolated on `aster/privacy-filter-smallest`; original dirty checkout untouched. No merge or push.

## 2026-10-26 — SHARP ONNX image-to-3D external adapter (isolated checkpoint)

- Experimental package `@lil-local/sharp-onnx` implements explicit WebGPU ONNX load, 1536px photo tensor ABI, Gaussian NDC→metric decoding (including covariance) and binary PLY output; exports `createSharpSubstrate` for existing Lil Local runner, not a product UI/server. Caller owns weights and camera calibration; Apple model license is research-only.
- 9 focused tests, typecheck, isolated packed external-consumer install/typecheck/build, 18 package boundaries, 73 qualification contracts, conversion validator and application builds passed. **Full `pnpm verify` remains blocked** at broad packed compatibility by TypeScript typed-array errors in unchanged image-embedding and Qwen3-TTS packages. No actual Apple SHARP model was loaded/run; no GPU/model quality claims.
- Local branch `aster/sharp-onnx-webgpu` isolated because the existing main checkout has unrelated changes to the same lockfile/compatibility files. No publish/push or changes to that main checkout. Full evidence: `docs/verification/2026-10-26-sharp-onnx-package.md`.


## 2026-10-18 — Rename Lil Tart / @litert-playground to Lil Local / @lil-local

- **Intent:** Give the browser model lab and reusable package family one coherent identity, `Lil Local`, namespace `@lil-local/*`, canonical repo `myrqyry/lil-local`; protect the established inference APIs and downstream apps.
- **Change:** Renamed all current workspace package manifests, peer links, source imports, examples, tests, packed external-consumer fixture, documentation, and visible browser title/guide copy. Lockfile regenerated with pnpm. The project remains a runtime qualification lab and local-first model launcher, not a Veyax cognitive organ.
- **Storage preservation:** Existing browser Cache Storage namespace `lil-tart-model-library-v1`, `/__lil_tart_model_cache__/` route and `x-lil-tart-*` cache metadata remain intentionally unchanged; so does default `createOpfsAssetStore('litert-playground')`. No model cache deletion/move is performed. A focused boundary regression test protects these identifiers.
- **Evidence:** New contract test initially failed under old root/package identity, then passed with the migrated namespace. `pnpm verify` exited 0, including real packed consumer installation + TypeScript build, tests, boundaries, conversion, qualification and production builds. No new model inference was claimed. Existing Vite chunk-size notices remain advisories.
- **Cross-repo boundary:** Older pinned commits retain their historical `@litert-playground/*` names. All Veyax Git dependencies must be switched as one revision to `@lil-local/*` with the matching new repo SHA; the updated worker's `advertisedEngines` and PCM audio prompt contract also differ from Veyax's historical pin. Preserve historical docs, existing local worktrees, and old consumer installs until their own explicit migration.


## 2026-10-18 — LiteRT-LM JS/WASM version alignment and real Chromium preload

- **Trigger:** After the 0.18.1 SDK upgrade, source inspection of its installed `dist/litertlm_web.js` revealed that upstream `LiteRtLm.DEFAULT_WASM_PATH` still targets `@litert-lm/core@0.12.1/wasm`. Upgrading the JS package alone did **not** verify which native WASM executed.
- **Fix:** The text pipeline, text worker and EmbeddingGemma 2 adapter now explicitly await upstream `getOrLoadGlobalLiteRtLm()` with matching `@litert-lm/core@0.18.1/wasm` assets, rather than accepting its old default. Each path accepts a caller-owned `wasmBase` so a consumer can serve its own same-version WASM/JS bundle for offline operation. A prior global module from a conflicting path is rejected, not silently reused. Added cross-package version-drift and failure/ordering regression coverage.
- **Verification:** Six new focused cases failed against the previous implementation and passed after correction; 64 focused direct/worker/embedding tests and 17 package-boundary tests passed. **Real Chromium** dynamically imported the installed 0.18.1 SDK, successfully initialized native WASM from version-pinned jsDelivr (observed both matching JS and WASM URL requests, zero page errors, no 0.12.1 requests). This is runtime-asset proof, **not** a real model generation or EmbeddingGemma vector test. Final full `pnpm verify` exited **0** on completed code, including all workspace typechecks and unit tests, boundary and runtime qualification tests, packed external-consumer checks, conversion assets and production builds. Only existing Vite chunk-size advisories remained.
- **Boundaries:** Default assets still require network unless caller self-hosts; a complete offline local-hosted browser smoke is not yet qualified. No model weights downloaded, no RikkaHub/phone/model-cache changes, no push. Detailed reproducible evidence: `docs/verification/2026-10-18-litertlm-wasm-alignment.md`.


## 2026-10-18 — LiteRT-LM JS 0.18.1 and EmbeddingGemma 2 opt-in adapter

- **Why:** Upstream shipped `@litert-lm/core@0.18.1` with the real JavaScript `EmbeddingEngine`; Lil Tart was still using `0.15.0`, and its existing `encoder` is a distinct Transformers/LiteRT path, not a LiteRT-LM model.
- **Implemented:** Upgraded and exact-pinned `@litert-lm/core@0.18.1` in text-gen; no change to its worker/streaming/cancellation logic. Added separate `@litert-playground/embedding-litertlm` package with public `createEmbeddingGemma2Substrate`, an immutable Hugging Face model-source revision URL, explicit model load, WebGPU-only text embedding through the actual `EmbeddingEngine.computeEmbedding` API, streaming HTTP fetch with AbortSignal, native model dispose delegated to the shared BrowserSubstrateRunner, 128/256/512/768-dimensional validated Float32 outputs, and explicit Google LiteRT task prefixes. Image/audio are not enabled in this slice.
- **Evidence:** `pnpm --filter @litert-playground/text-gen test` passed 74 cases, text-gen and playground typecheck passed, `pnpm --filter @litert-playground/embedding-litertlm test` passed 7 adapter contract cases, and package typecheck passed. The new adapter passes the **packed external-consumer compatibility** harness (pack, install, typecheck, Vite build) without any model fetch or execution. Final full `pnpm verify` exited **0** on the finished code: all workspace typechecks/tests, WebGPU selection tests, package boundaries, packed external-consumer compatibility, conversion assets, runtime qualification tests and production builds passed. Vite chunk-size notices remained advisory. A red boundary test caught pnpm saving `^0.18.1`; both the text-gen manifest and lockfile now specify **exact** `0.18.1`.
- **Boundaries:** Installed upstream npm declarations/implementation and Google docs are real, but the embedding test engine is controlled/mocked. The pinned Hub revision was observed through metadata, not file-hashed. Neither EmbeddingGemma 2 weights nor new-version text-generation models were run on a real GPU. No changes to RikkaHub Local, its Android 0.18.0 dependency, or user phone models. Keep distinct from durable browser/model qualification; see `docs/verification/2026-10-18-litertlm-0181-embeddinggemma2.md`.


## 2026-10-18 — D1 ONNX disposal and error fidelity

- **Why:** The previous cooperative cancellation work awaited cleanup through the shared substrate runner, but the D1 ONNX implementation still handled concurrent direct `dispose()` callers independently. A second caller could resolve while the first was still awaiting `InferenceSession.release()`, or even report success after that release failed. Also, when a load/ABI failure and resource release both failed, the catch path replaced the primary error with the release error.
- **Change:** D1 ONNX now stores and reuses one disposal promise, giving every concurrent and later caller the same eventual success or failure, and releasing the session exactly once. Dual load-and-release failures retain both original errors in `D1LoadCleanupError.errors` instead of dropping either. The error class is internal and compatible with this repository's ES2020 TypeScript target; no SDK update or global runtime changes.
- **Evidence:** Added three D1 runtime regression tests, first confirmed two failing against the old disposal path and one against error masking. Focused D1/runtime/substrate tests passed **17/17**, and the D1 package typecheck passed. Full `pnpm verify` exited **0** on the finished code, including all workspace typechecks/tests, package boundaries, packed-consumer compatibility, conversion assets, runtime qualification, and production builds. Existing Vite chunk-size warnings were advisory.
- **Scope:** No model weights fetched, no inference or actual WebGPU release measured, no changes to model provenance/quantization, no RikkaHub/phone changes. This is deterministic lifecycle correctness rather than a claim of better generation performance.


## 2026-10-18 — Cooperative browser substrate load cancellation

- **Objective:** Strengthen the shared `BrowserSubstrateRunner` and the real D1 ONNX integration without conflating runtime identity, model format, requested GPU backend, or independent model inference evidence.
- **Implementation:** Extend the public runner's optional load/create contract with an `AbortSignal`. Pre-aborted loads never call the engine; in-flight abort suppresses late progress, forwards the signal to the loader, and disposes a late-returned instance before rejecting with `AbortError`. A cleanup failure is propagated, moves the runner to a non-reusable `failed` state, blocks new loads, and remains observable to `dispose()`. Existing no-signal callers continue unchanged. D1 checks cancellation between tokenizer and graph stages, after ORT graph creation, and on ready; the catch path releases any created session. Its UI offers **Stop loading**, retains actual cleanup/loader errors, and aborts/disposes pending sessions on unmount.
- **Evidence:** Focused tests in `inference-core` include pre-abort, forwarded signal, late cleanup, late progress, and failed cleanup. New D1 runtime tests use controlled tokenizer/ORT test doubles and demonstrate that a graph returned after cancellation is released and never reported ready; the D1 adapter test verifies signal forwarding. TypeScript checks and production build passed. Real headless Chromium against the production preview opened the D1 panel, exposed the Stop control during load, and after a deliberately blocked/failed model request reported **Load stopped; model not loaded**; recorded zero model-weight requests and zero page errors. This is a **UI/lifecycle smoke only**, not real ONNX graph inference or transport-abort proof.
- **Limitation:** ORT `InferenceSession.create` receives URLs and currently offers no attached `AbortSignal` through this adapter. A download/compile already inside ORT can continue until its call returns; the runner guarantees late-result disposal, not immediate network interruption. Existing LiteRT inference code, RikkaHub, installed model caches and phone data were not changed. Final `pnpm verify` passed exit 0 on the completed implementation, including typechecks, package/unit tests, WebGPU-selection tests, package boundaries, packed external-consumer compatibility, conversion-asset and runtime-qualification tests, and production builds. Existing Vite chunk-size warnings were advisories. Work remains local only; no remote push or mobile installation.

## 2026-10-18 — LiteRT Community read-only discovery (local checkpoint)

- **Goal:** Extend Lil Tart's existing curated LiteRT adapter library with live browsing of the full `litert-community` Hugging Face organization, without treating arbitrary published artifacts as runnable models.
- **Implemented:** The playground's `Explore community` mode uses the public Hub model-metadata listing (40 per page, explicit server-side search, manual pagination). Parsed IDs are constrained to the actual organization and canonical repo slugs; duplicate/private/invalid entries are filtered. A repository match is made only against existing model-specific LiteRT adapter source URLs, and the view shows that adapter's own verification status rather than manufacturing runtime support. `Inspect` navigates to the existing adapter panel. Community-only entries link to their source; no weight download, installation, cache write, or runtime activation is introduced.
- **Evidence:** New pure unit tests were run red/green for source URL construction, ID filtering, deduplication, invalid/unknown metadata, and exact adapter matching. Focused tests, `pnpm --filter playground typecheck` and production Vite build passed. Final `pnpm verify` passed exit 0 after the display correction for unknown download counts; existing package, compatibility, boundary, qualification and production-build gates were included. Headless Chromium against the production preview loaded live community data, successfully searched for `moonshine`, and handed `moonshine-tiny` to its existing adapter panel. The observed flow made **zero weight-file requests** and had **zero page errors**.
- **Boundaries:** RikkaHub Local and phone model storage were untouched. Catalog discovery is not LiteRT/LiteRT-LM execution qualification; the RikkaHub separate LiteRT-Models gallery and EmbeddingGemma integration remain independent. No publication/push. Remaining: separately qualify any model needing a new adapter before making it executable. Query URL pagination and metadata parsing have pure unit coverage; clicking Load more and the failure/retry UI have not been exercised in a browser. The initial search/Inspect path was exercised against the live API.


## Current checkpoint — five-commit Lil Tart verification — 2026-10-17

- Working branch `master` began clean, **five commits ahead** of `origin/master` at `2f74ea1`, with remote `ebd3c8d`. The five commits add an opt-in D1 ONNX decision model, a verified MoGe-2 geometry package, an engine-neutral browser substrate lifecycle, and an independent WebGPU matmul feasibility probe. No other projects, model stores, or live services were edited.
- **Current-source gate:** `pnpm verify` exit **0** (complete repository typecheck, package/example tests, 3 WebGPU selection tests, 16 package-boundary tests, external-consumer package compatibility, conversion-asset checks, 73 runtime-qualification tests, production builds). Chunk-size warnings were advisories, not errors. This gate ran before only prose-level doc/checkpoint edits.
- **Fresh browser/hardware checks:** `pnpm probe:webgpu-matmul --size 128` exit **0** on NVIDIA/Turing WebGPU: both workgroup variants correct vs CPU (maximum absolute error ~9.09e-7), selection **null / indistinguishable**. This is a separate WGSL benchmark; no LiteRT/ONNX inference improvement claimed. Isolated Chromium loaded the freshly built playground and opened the D1 model panel without errors; **zero external model-weight fetches before Load**. The D1 model was **not downloaded/loaded/run again** on this checkpoint, so the post-substrate-migration browser inference result remains unqualified. Existing Oct 12 D1/Q8 real WebGPU-requested inference and MoGe-2/SwiftShader execution are historical evidence, not today's proof.
- **Publication boundary:** public repository; inspected tracked five-commit additions (2,271 added lines) for common credential patterns, tailnet addresses and personal host paths with no matches. The global push hook initially reported terminology in docs rather than actual implementation gaps; three documentation-only clarifications preserve the meaning and now allow the hook to pass **without bypassing it**. No dependencies or production logic changed in this audit.
- **Remaining gates:** fresh D1 browser inference **through the new substrate wrapper**, multi-request correctness/latency/VRAM, proper WebGPU execution-provider accounting, new MoGe real NVIDIA path, and actual downstream Veyax import against one synchronized Git revision. Prior isolated D1 load/run and MoGe SwiftShader results must not be silently extended.
- Private verification receipt: `.meristem/local/verification-2026-10-17.md`. Portfolio receipt: `~/loam/PROJECT-INVENTORY.md`.

## Current follow-up — WebGPU kernel tuning feasibility — 2026-10-16

A bounded real-NVIDIA WebGPU matmul probe now compares 8x8 and 16x16
workgroups, verifies every output against CPU results, and refuses
to choose a configuration without at least a 10% median advantage
and nonoverlapping measurement samples. It was executed at
128, 256 and 512 square matrix sizes on NVIDIA Turing/Chromium.
All runs passed numerical checks, with **no reliable winner**.
The normal pnpm verify gate includes three pure selection unit tests.
This research cutting **does not intercept or accelerate LiteRT/ONNX**
kernels; their runtime APIs do not expose direct replacement hooks.
No inference/runtime/backend behavior was modified.
Reproduction and actual measurements:
docs/verification/2026-10-16-webgpu-kernel-tuning-feasibility.md

## Current follow-up — multi-substrate browser inference — 2026-10-13

Lil Tart now has an engine-neutral `BrowserSubstrateRunner` published through
`inference-core`; the existing D1 ONNX Runtime Web panel delegates to the
real ONNX model via an adapter to that interface. Established LiteRT model
packages and the managed LiteRT runner remain unchanged. The packed external
consumer imports the new API. Full `pnpm verify` passed, and focused runner
and ONNX-bridge tests passed.

**Unfinished:** Fresh real-browser D1 smoke after the UI wiring change, runtime
receipts common across engines, and opt-in qualification of additional
substrates (GGUF etc.). Structural typecheck and simulated adapter tests
must not be described as live model proof. Details: `docs/web-substrates.md`
and `.meristem/DECISIONS.md`.


## Current follow-up — MoGe-2 shared geometry extraction — 2026-10-12

The real MoGe-2 LiteRT browser proof has been extracted from Veyax into Lil
Tart's existing `depth-estimation` package (the old Depth Anything 3
pipeline remains intact). Both canonical relative DepthFrame and richer affine
point/normal/mask/scale outputs are exposed through the public entrypoint.
Independent Lil Tart WebGPU browser execution passed, and the complete
`pnpm verify` gate passed, including packed external-consumer tests.
Detailed evidence, limitations and model revision are recorded in
`docs/verification/2026-10-12-moge2-port.md`.

**Unfinished:** Veyax still uses its verified local adapter. It cannot
consume this unpublished Lil Tart revision from its Git-pinned dependencies
until that revision is available remotely and all Lil Tart packages are
advanced to the same SHA. No push or downstream dependency rewrite was done.
This is a consumer migration boundary, not evidence that Veyax has been
rewired. HyperShaderFX native C++ also remains separate.


## Current follow-up — D1 Omni Q8 ONNX WebGPU text qualification — 2026-10-12

Lil Tart now includes an explicitly user-loaded ONNX Runtime Web decision path, independently of its existing LiteRT runtime. The pinned D1 Omni 600M Q8 text graph executes choice and yes/no questions, reproduces the source prompt marker contract and score temperatures, and reports raw uncalibrated outcomes. No weights are downloaded by navigating to the panel. The package is consumed through its public entrypoint and the packed-consumer lane. The prior runtime-receipt/provenance work remains intact.

**Observed evidence:** Real NVIDIA/Turing headless Chromium with Vulkan/WebGPU requested successfully loaded the model and returned a choice decision. This is one real inference request, **not** a benchmark, numerical parity result, or full GPU-node assignment proof (ORT warned some nodes were assigned elsewhere). See \`docs/verification/2026-10-12-d1-webgpu-text.md\`.

**Unfinished:** Image/audio ONNX graph composition, quantified parity against native GGUF D1, repeated latency/VRAM calibration, browser cache management, cancelable model download, and Veyax consumption. Keep this as an opt-in lab until those paths are actually verified.

## Current follow-up — runtime evidence export and model provenance — 2026-10-10

Following `77fe3af` (the pushed opt-in inference priority scheduler), the
playground now has a versioned, explicitly session-scoped runtime proof JSON
export and a selected-model disclosure for the adapter's recorded verification
status, backends, date, note, and evidence reference. Neither operation promotes
an adapter's durable qualification or claims that a model passed browser
inference without fresh runtime evidence.

The export is offered only when `useModelRunner` has an actual non-null
`RuntimePathProof`, produced from inference telemetry. Download names are
sanitized, the anchor is attached/removed, and object URLs are revoked after
a grace period rather than before slower browsers begin their downloads.
The UI labels recorded evidence without treating the `registered` status as
verified. See `.meristem/strategy/lil-tart/10x/session-1.md` for the review and
scope decisions.

Verification: focused playground typecheck and 168 tests passed on the updated
working tree. Full `pnpm verify` exited 0 on this source tree on 2026-10-10,
covering typechecking, package/example tests, boundary checks, packed-consumer
compatibility, conversion validation, qualification contracts, and builds.
Actual browser download/inference smoke has **not** been performed; the test
suite does not establish real device/model behavior. No downstream consumer
integration has been demonstrated.

## Historical state — Verify-gate hardening — 2026-10-06

At that checkpoint, the checkout was clean `master` at `fda8cda`, matching `origin/master`.
This commit follows PR #15's merge (`1e72863`) and contains the Verify-gate
follow-up: Node 22.16-compatible example config loading, qualification rejection
of failed numeric parity, and complete pinned-recipe validation with mutation
coverage.

Fresh `pnpm verify` passed on the exact pre-commit tree with Node 22.16.0 and
pnpm 11.17.0. Detailed diagnosis, commands, counts, and evidence limits are in
`docs/verification/2026-10-05-pr15-verify-followup.md`. Hosted CI status for
`fda8cda` and the current status of PR #15 review threads have not been checked
in this session. No new browser model-inference or numerical-reference evidence
was recorded; candidate models remain browser-unverified.

No follow-on implementation objective is stated in this session. Older work
records below are retained as history and are not current branch/status claims.

## Historical follow-up — 2026-09-28, integrated onto `2200684`

This section records the earlier PR #10 continuation. The
remote PR #10 branch was checked again and still pointed to `a3dfeb4`. Its Vercel
comment reported Ready at 07:48 UTC, newer than the 05:27 snapshot in the task.
The user subsequently authorized committing and pushing. Before delivery, the
remote advanced to `2200684` via `632ba43`; the follow-up is rebased on those
commits, preserving their caller-signal cancellation checks and probe metadata.
Final combined validation and delivery are recorded in the follow-up report.

- The original four-item review was already addressed in `c4a606e`: scoped
  removal, pruning documentation, cancellable streamed checkpoint fetch, and the
  public options re-export. Preserve those changes and the scoring export.
- New patch: cache membership uses the original asset path resolved against the
  current base. All listing/deletion/pruning paths share that predicate. Legacy
  entries are recovered from registered paths or a matching recorded base;
  otherwise they remain visible as **unverified** and are never scoped-deleted or
  automatically pruned. This replaces the historical prefix/base-only rules.
- New patch: late tokenizer results remain local until load succeeds; disposed
  pipelines reject later loads and stay disposed after in-flight inference.
  Text disposal attempts both native cleanups, reports failures, and preserves
  cancellation with the cleanup error as its cause for late engines.
- Text checkpoints still bypass Cache Storage to retain real response streaming.
  The LFM UI now states this explicitly and never presents text weights as a
  model-library cache hit. A streaming persistent-cache implementation remains
  separate work; no full checkpoint buffering was introduced.
- 20 added regression cases; 75 initial focused tests pass. After integrating
  the newer commits, 144 focused tests across 26 files pass. The first 16 new failure
  cases were demonstrated against the unchanged production baseline.
- Final combined `pnpm verify`: exit 0 (typecheck, 355 package/example tests, 15 boundary
  tests, 10 compatibility packages, 65 qualification tests, and builds).
  Fresh browser qualification is blocked by a missing Chromium executable and
  failed browser downloads; historical browser results are not a fresh pass.
- Subsequent pasted review: the qualification probe now calls production
  `resolve()` with progress-triggered abort, not unused `stream()`. Its fixture and
  HTTP server are created only when the case invokes the probe; imports/help and
  unrelated cases are side-effect free. Missing/read-failed files are handled,
  partial startup closes all acquired resources, and the ignore contract checks
  Git itself. The unused case re-export is removed.
- Qualification coverage is now 65 tests across 22 files, including 11 new cases.
  Real Node HTTP checks pass and a lost-signal negative control consumes all
  24 MiB and fails as intended. A fresh browser attempt remains blocked by the
  missing Chromium executable, but exits without leaking a server.
- The old Chromium observation exercised only `stream()`; it must not be cited as
  proof of the corrected production resolver or complete pipeline cancellation.
- Detailed validation and remaining limitations:
  `docs/verification/2026-09-28-pr10-followup.md`.

## Historical continuation — conversion qualification forward-port — 2026-10-05

The old `aster/litert-upstream-hardening` branch was 115 commits behind current
`master`, while its provenance and MediaPipe work had already landed separately.
The branch is being rebuilt from current `master` with only the still-distinct
conversion/qualification work preserved.

- Add a pinned conversion-evidence lane sourced from
  `john-rocky/LiteRT-Models@460d52221c22388b6a9a8e8a44b61dde72976b4e`.
- Keep Android conversion/delegate evidence separate from Lil Tart browser
  qualification; candidate models remain browser-unverified until observed here.
- Add numeric-parity evidence to qualification results so compile/run success
  cannot silently stand in for output correctness.
- Add deterministic conversion-asset validation to the authoritative
  `pnpm verify` gate.
- Preserve current master runtime/provenance/MediaPipe implementations rather
  than replaying their older copies from this historical branch.

Fresh verification against the rebuilt tree is established: `pnpm test:conversion-assets`
passes for 10 candidates and 2 pinned recipes; the qualification suite passes
69/69 tests; and full `pnpm verify` exits 0 across typecheck, package/app tests,
15/15 boundary tests, all 10 packed-consumer compatibility rows, conversion
validation, qualification, and production builds.

## Historical notes through `a3dfeb4`

Retained for decision history. Statements below about dirty state, branch distance,
uncancellable transfers, prefix/base-only cache classification, and old verification
counts describe earlier stages and are not the current status.

Subject: `fix/consumer-runtime-integration` — downstream consumer hardening and
the retrieval scoring entrypoint.

## Outcome

A downstream application can consume Lil Tart inference packages by Git SHA
through their public entrypoints, without hitting app-origin or stricter-TypeScript
breakage, and the retrieval scoring surface is a deliberate, documented choice
rather than an accidental export.

## Why it matters

This branch defines the supported downstream surface other apps are expected to
depend on. Anything published here becomes a contract: once a consumer can reach
it, removing or reshaping it is a breaking change under the project's single-revision
policy.

## Current state

**Branch context** (committed, and already pushed to
`origin/fix/consumer-runtime-integration`; not yet in `master`):

- `32e666a` *fix: harden shared consumer runtime paths* — resolve LFM pipeline
  model assets from Hugging Face instead of the app origin; remove unused
  retrieval state that broke stricter downstream TypeScript consumers.
  Touches `apps/playground/src/components/LfmPipelinePanel.tsx`,
  `packages/retrieval/src/colbert.ts`.
- `8c9cb8b` *feat: expose retrieval scoring subpath* — add
  `"./scoring": "./src/scoring.ts"` to `packages/retrieval/package.json` so
  consumers can import pure scoring without loading the ColBERT package entrypoint.

**Current working tree (dirty, uncommitted):** the branch commits above were
complete, but three defects and one missing contract surfaced in review. All are
now closed in the working tree.

Subpath coverage (from the first review round):

- `tests/fixtures/external-consumer/src/index.ts` — imports `maxSim` and
  `rankDense` from `@litert-playground/retrieval/scoring`, so the packed tarball
  must resolve the subpath for `pnpm test:compatibility` to pass.
- `tests/package-boundaries.test.ts` — new test
  *"keeps the retrieval scoring subpath a verified public entrypoint"*: pins the
  `exports` map, reads the subpath target so a rename breaks the suite, and
  asserts `scoring.ts` never imports `./colbert` or `@huggingface/transformers`
  (that independence is the subpath's entire reason to exist).
- `docs/package-revision-policy.md` — states that subpath entrypoints are
  supported on the same terms as `.`, and lists `@litert-playground/retrieval/scoring`.

The base-swap follow-ups:

- `packages/text-gen/src/litertlm-pipeline.ts` — the real bug. `load()` passed the
  manifest's repo-relative path straight to `Engine.create`, bypassing
  `context.assets` entirely, so all four `kind: 'text'` models in the LFM panel
  still resolved against the app origin and 404'd. The branch's stated purpose was
  only half-delivered. Now resolved to an absolute URL from an explicit
  `modelBase` option.
  - **First attempt was wrong and was corrected twice.** Routing through
    `context.assets.stream()` fixed the 404 but buffered the whole checkpoint
    (that resolver awaits `resolve()` first). The absolute-URL form fixed the
    memory profile but not cancellation. The final form fetches the URL *in the
    pipeline* with an `AbortSignal` and hands the engine `response.body`, which
    `modelToStream` passes through untouched. Upstream evidence for each step:
    `modelToStream` accepts a `ReadableStream` as-is; its own string fetch uses
    `credentials: 'same-origin'` and **no signal**, which is why the pipeline has
    to fetch; `EngineSettings` has no signal field, so the engine could not have
    been handed one. Memory and streaming behaviour are unchanged, because from
    `modelToStream` onward a string and a stream are handled identically.
  - Cancellation is wired from both directions: `context.signal` (which the panel
    supplies) aborts an internal controller, and `dispose()` aborts it too, so a
    consumer that wires no signal can still cancel by disposing. A cancelled load
    leaves status `idle` rather than latched to `error`, so it stays retryable.
  - Trade-off accepted: text models are fetched by the pipeline rather than the
    engine, so they do **not** participate in Cache Storage. Before this branch
    they were broken (404), so nothing working was lost. Gaining caching without
    buffering would mean making the playground resolver tee a real `fetch` body
    instead of wrapping `resolve()`, which is a larger change and is not done.
- `apps/playground/src/modelStorage.ts` — `listStoredModels(base)` counts only
  entries under that base, and `pruneSupersededModelCaches` reclaims entries
  orphaned by a base change. Cache keys embed the resolved absolute URL, so every
  pre-move entry was unreachable while still being counted.
  - `belongsToBase` compares origins and path prefixes **structurally**. A lexical
    `startsWith` was wrong for any base without a trailing slash: base
    `https://huggingface.co/litert-community` claimed
    `.../litert-community-evil/...`, and since `prune` deletes, that was a data-loss
    path reachable from free-text user input. The first version of this guard was
    only passing its test because the Hugging Face base happens to end in `/`.
  - `listOrphanedModels(base)` surfaces entries written under another base. In the
    LFM panel the base is a constant that will never change back, so pruning is
    right there. In `ModelRunner` the base is free-text user input, so pruning on
    change would irreversibly delete multi-gigabyte downloads the user may switch
    back to; those are surfaced with a per-model Remove instead, reusing
    removal scoped to the orphan set.
  - Removal is base-scoped: `removeStoredModel(modelId, base)` deletes only the
    live copy and `removeOrphanedModel(modelId, base)` only the previous-base
    bytes. A model id can hold entries under both bases at once, so a single
    unqualified delete behind a "previous base" button would destroy the live copy
    too. The orphan button also deliberately does not unload the running model,
    since the entry it removes is not the one in use.
- `packages/text-gen/src/index.ts` — `LiteRtLmTextPipelineOptions` is re-exported
  next to `LiteRtLmTextConfig`. Without it a consumer can pass `{ modelBase }` but
  cannot name the type, so the supported surface had a hole exactly where the
  packed-tarball harness should police it; the external-consumer fixture now
  imports and uses the type, so a future removal fails `test:compatibility`.
- `apps/playground/src/modelStorage.test.ts` — covers base filtering, the
  prefix-collision case that the first guard missed, the slash-less base, orphan
  listing being non-destructive, and both base-scoped removal directions.
- `tests/package-boundaries.test.ts` — added *"keeps model-cache removal
  base-scoped in storage and at the call site"*, which asserts the storage API is
  base-scoped and that the orphan row routes through the orphan variant. The unit
  tests cannot see component wiring (the playground test environment is `node`,
  with no DOM), so this is the only thing that catches the UI regressing to an
  unqualified delete.
- `apps/playground/src/components/ModelRunner.tsx` — lists orphans alongside the
  current base and renders them with per-model removal.
- `apps/playground/src/components/LfmPipelinePanel.tsx` — `MODEL_BASE` constant so
  the resolver and the text pipeline's `modelBase` cannot drift; comment recording
  that omitting `assetBase` moves the LiteRT WASM host to the pinned jsDelivr CDN
  (`apps/playground/public/` has no `wasm/` dir, so origin-relative was dead).
- `CHANGELOG.md` — entries for every change on this branch, which previously had none.
- `.meristem/` + bounded `.gitignore`/`AGENTS.md` blocks from substrate init.

Nothing is staged. No stashes.

**Base:** `master` is at `c3bd5a5` (merge of `aster/downstream-inference-surface`).
The branch is two commits ahead of it.

## Locked decisions

- The new retrieval entrypoint is a **subpath export** (`./scoring`), not a new
  package and not a deep `src` import. The revision policy explicitly forbids
  consumers importing from a package's `src` directory.
- Model assets for the LFM pipeline resolve from Hugging Face, not the app
  origin, so the app and package consumers agree on where weights come from.
- Repository is `myrqyry/lil-tart`; package namespace stays
  `@litert-playground/*`. Do not "fix" the mismatch.

## Constraints / do-not-regress

- `pnpm verify` is the pre-merge gate. Do not bypass the pre-push hook.
- Boundary tests enforce that `inference-core` stays independent, examples consume
  public entrypoints, and the playground depends on `runtime-litert`. A new
  entrypoint must not weaken these.
- Anything claimed as supported downstream surface must be packable and
  consumable by `pnpm test:compatibility`, with no `workspace:*` leakage in packed
  metadata.
- Keep the supported packed surface and `CHANGELOG.md` consistent with what
  packages actually export. The project keeps a live `[Unreleased]` section;
  shipped public API changes belong there.
- Stage by explicit path, never `git add .`.

## Current objective

Land the branch. The downstream contract for the `./scoring` entrypoint is now
decided, tested, and documented, so the remaining work is delivery:

1. Review the working-tree changes above and commit them by explicit path
   (never `git add .`).
2. Mention the `./scoring` export in the PR description so consumers know it
   exists — the one part of the review feedback not satisfiable from the repo.
3. Get the branch into `master` once CI is green.

## Verification

Established against the current tree, not inherited from earlier commits:

- `pnpm verify` — exit code 0. All six stages green: typecheck, per-package
  tests (inference-core 33, text-gen 22, encoder 11, retrieval 11, playground 141),
  boundaries 15/15, compatibility 10/10 packages, qualification 20 files, build.
- `@litert-playground/retrieval` compatibility row: `pass pass pass pass`
  (import, peers, typecheck, build) with the subpath import live.
- **Mutation check.** Three deliberate regressions were introduced and reverted;
  each was caught rather than passing silently:
  1. Renaming the export key to `./scoring-typo` — boundary test failed on the
     `exports` match, and the compatibility harness failed with
     `TS2307: Cannot find module '@litert-playground/retrieval/scoring'`.
  2. Restoring the old raw-path hand-off in text-gen — all five text-gen tests failed.
  3. Deleting the `belongsToBase` guard in `listStoredModels` — the storage tests failed.
  4. Reverting `belongsToBase` to a lexical `startsWith` — only the new
     sibling-path test failed, which is how the first guard's weakness was proven.
  5. Re-buffffering the model through `context.assets.resolve` in text-gen — five
     text-gen tests failed, including the one asserting the resolver is never called.
  6. Letting the `catch` stamp `idle` over a disposed pipeline — the dispose tests failed.
  7. Dropping the already-aborted signal check — the pre-aborted-signal tests failed.
  8. Deleting the abort line the LFM cancellation guard protected — **the old
     assertion passed it**, which is how that guard was proven unfalsifiable; the
     fixed assertion fails.
  9. Dropping the signal from the runtime resolver request in the browser probe —
     the qualification case reported the full 25,165,824 bytes arriving.
  Each mutation was reverted and the suite re-confirmed green.

Browser runtime evidence is now partially established. The `pipeline-load-cancellation`
qualification case runs in headless Chromium and streams a real cross-origin asset
through the shared resolver, observing that an abort stops the transfer well short of
24 MiB and surfaces `CANCELLED`. Dropping the signal reproduces the defect exactly
(full 25,165,824 of 25,165,824 bytes), so the case is falsifiable rather than
decorative. Record: `docs/verification/2026-09-28-model-load-cancellation.md`.

Still not established: `Engine.create` consuming a streamed body, which needs a real
`.litertlm` checkpoint, and any actual text generation in a browser. The text-gen
change is verified at the type/API boundary, by unit tests, and now for the transfer
half in a real browser; the engine half is not.

## Open uncertainty

- Whether `./scoring` duplicates logic still reachable from `.` is now
  answerable but unchecked: `src/index.ts` re-exports all five scoring
  functions, so the subpath and `.` currently overlap completely. That is
  harmless, but the subpath's independence is only meaningful if the two are
  expected to diverge. Worth a deliberate decision at some point; not blocking.
- `pruneSupersededModelCaches` is called only where the base is a constant (the LFM
  panel). It is deliberately not called from `ModelRunner`, where the base is
  free-text user input. If two panels ever share a `modelId` under different bases,
  the LFM panel's prune could reclaim the other panel's copy. Not currently the case;
  worth confirming it stays that way.
- **Text-model downloads cannot be cancelled, and this is pre-existing, not a
  regression from this branch.** A review claimed the absolute-URL fix dropped an
  abort path. Checked before accepting it: upstream `EngineSettings`
  (`@litert-lm/core@0.15.0`) is `{ model, backend?, mainExecutorSettings?,
  benchmarkEnabled? }` — no signal field, so the engine cannot be handed an
  `AbortSignal` at all. And the LFM panel never passes `signal` to
  `createLiteRtRuntime` (`LfmPipelinePanel.tsx:106`), so `context.signal` was
  `undefined` in the discarded resolver version too. Before this branch the text
  path 404'd outright, so no working cancellation was lost. The same gap applies
  to encoder/colbert, whose `context.liteRt.loadModel` also sees an undefined
  signal. Fixing it means the panel owning an `AbortController`, passing it into
  `createLiteRtRuntime`, and the pipeline consuming it — via a new engine option
  upstream, or by fetching the URL in the pipeline and handing `response.body` to
  the engine. That is a cross-cutting design change, not a review fix, and is
  **not done**. Worth deciding deliberately.
- Cache membership is decided by the base an entry was **written with** (a
  `x-lil-tart-base` header), not by a path prefix. Prefix matching is too loose once
  bases are nested: an entry cached at `https://host/v1/model.tflite` does sit under
  `https://host/`, but after widening the base the resolver asks for
  `https://host/model.tflite` and never reads it, so counting it would reintroduce
  the exact "stored locally but re-downloads" lie this work removed. Entries written
  before the header existed carry no recorded base and fall back to the structural
  prefix comparison, so no existing cache is invalidated on upgrade.
- A `dispose()` that lands while `Engine.create` or `loadModel` is still pending
  cannot cancel that work. All three pipelines carry a `loadToken`, and `dispose()`
  also sets an explicit `disposed` flag *before* aborting, so:
  - a result that arrives late is released, not published (text generation deletes
    the orphaned engine outright because WASM engine memory is not tracked by the
    runtime; encoder and colbert only skip publishing, since the runtime owns and
    disposes the compiled model);
  - a `load()` abandoned by disposal **rejects** with `CANCELLED` rather than
    resolving quietly, because the caller asked for a model and did not get one;
  - the `catch` reports `disposed` when disposal won the race, instead of stamping
    `idle` over an already disposed pipeline;
  - a tokenizer fetch completing after disposal does not repopulate state;
  - **both** cancellation routes are checked at every publication point. Disposal
    and a superseding load bump `loadToken`, but a caller aborting `context.signal`
    only flips `controller.signal` — checking the token alone let a cancelled load
    finish with a live engine and `status: 'ready'`, because a compile already under
    way cannot be aborted. An already-cancelled pipeline is now also never started:
    the model load is skipped entirely rather than started and cancelled.
  Awaiting the in-flight `load()` inside `dispose()` was considered and rejected:
  guarding the assignment sites makes correctness independent of ordering, and
  blocking `dispose()` on a multi-hundred-megabyte compile would be its own problem.
- An already-aborted `context.signal` is now handled explicitly in all three
  pipelines. Subscribing to a settled signal never fires, so `addEventListener`
  alone was an elaborate way to do nothing: a caller that cancelled before dialling
  got a full checkpoint download anyway. The playground masked this because
  `RuntimeContext.signal` is the same object the runtime holds, but a consumer
  implementing the public contract with its own signal would have hit it. This
  mirrors `startPendingLoad` in `runtime-litert/src/context.ts`, which already had
  the `if (runtimeSignal.aborted)` branch.
- **A test of mine could not fail, and it is worth recording as a lesson.** The LFM
  cancellation guard compared two `indexOf` offsets. A missing needle returns `-1`
  and `-1 < anything`, so deleting the very line the test existed to protect still
  passed. Confirmed by mutation: with the old assertion the suite reported 15/15
  green after removing the abort line; with the fix it fails. Any offset comparison
  must assert both sides are non-negative first, and the slice should be bounded to
  the function under test.
- Rejected after checking, worth recording so nobody re-raises it: the model runtime
  **is** recreated when `modelBase` changes. The `[modelBase]` effect in
  `useModelRunner.ts:322` returns a cleanup that runs on every base change and does
  `runtimePromiseRef.current = null` before disposing, so the next `ensureRuntime()`
  builds a fresh runtime with the new base. A review claimed the memoized runtime
  survives the change; it does not.
- Text models bypass Cache Storage entirely (see the trade-off note above). This
  is also why `storedInfo` is null for them in the LFM panel: the panel renders
  "Stored locally · N MB" only when `storedInfo` is set and otherwise labels the
  action "Download & load", so the readout is truthful rather than misleading.
  Restoring caching needs the playground resolver to tee a real `fetch` response
  body into the cache rather than wrapping `resolve()`, which would give every
  consumer caching without reintroducing a full buffer.
- The tokenizer load in encoder and colbert is **not** cancellable:
  `AutoTokenizer.from_pretrained(repoId)` exposes no signal, so that fetch still runs
  to completion after a cancel. Only the model load is covered. Closing it needs an
  upstream transformers.js change.
- `LiteRtRuntime` in inference-core was widened to
  `loadModel(path, options?: { signal?: AbortSignal })` (plus positional `signal` on
  `loadNpy`/`fetchBuffer`) to match what the runtime already implemented. The old
  narrow signature hid a real capability, so encoder and colbert could not pass a
  signal even though `ManagedLiteRtRuntime.loadModel` accepted one and
  `startPendingLoad` already honoured per-call signals. Additive and optional, so no
  consumer breaks.
- Worth knowing, because it changes what the panel fix alone would have achieved:
  the runtime links `LiteRtRuntimeOptions.signal` in `startPendingLoad`, so once the
  panel passed `signal: loadAbort.signal`, encoder and colbert model loads were
  *already* cancellable through the runtime-level signal. The pipeline-level
  controller added on top is what makes `dispose()` alone sufficient for a consumer
  that wires no signal, and keeps the three pipelines consistent.
- The tokenizer path in encoder/colbert calls `AutoTokenizer.from_pretrained(repoId)`,
  which resolves against the Hugging Face hub directly and therefore ignores a
  custom model base. Pre-existing, not introduced here, and out of scope.
- Whether `packages/retrieval` peer/version metadata needs any change for a
  second published entrypoint (currently `0.1.x` peer on `inference-core`).
  Judgment says no, since `./scoring` imports only types from `inference-core`.
- `docs/superpowers/` is git-ignored and holds local plans/specs; it may contain
  intent for this work that is deliberately not shared. Not inspected.
- `AGENTS.md` and `docs/package-revision-policy.md` still duplicate conventions,
  workflow rules, and the nested-agent-file list. Now that the subpath rule had
  to be written into the policy doc, the duplication cost something concrete.
  Collapsing them into one authority is a real decision still unmade.

## 2026-10-19 — Native LiteRT-LM backend enum repair for downstream app

- **Trigger:** Creator's Insight Engine loaded 0.18.1 WASM on real NVIDIA/Turing WebGPU and completed SHA-256-verified Qwen3 0.6B model download, then failed with `wasm.ModelAssets.create is not a function`.
- **Root cause:** `LiteRtLmTextPipeline` forwarded the generic backend string `webgpu`, but SDK Engine.create requires numeric `Backend.GPU_ARTISAN=2`; the string selected a wrong non-streaming branch. `wasm` similarly needs `Backend.CPU_ARTISAN=1`.
- **Fix:** Translate at SDK boundary without changing the consumer API; regressions assert numeric native values. Removed three unused members reported by strict downstream TS source consumption.
- **Evidence:** RED 3/80, GREEN 80/80 focused text package tests, package typecheck and full `pnpm verify` exit 0. Verified native checkpoint inference still pending rerun in the downstream app. See `docs/verification/2026-10-19-litertlm-numeric-backend.md`.

## 2026-10-19 — Qwen3 0.6B WebGPU through Transformers.js

- **Evidence:** Actual Chromium on NVIDIA Turing loaded the pinned ONNX Qwen3 0.6B checkpoint with Transformers.js 4.2.0 and generated an assistant response. The prompt's 48-token budget was consumed by its thinking content, so this is model-execution proof, not grounded creator-summary quality proof.
- **SDK limitation:** LiteRT-LM 0.18.1 native bindings failed `wasmEngineSettings.delete is not a function` after the generic backend mapping fix. Its JS and WASM exports disagree; no native LiteRT-LM text inference was claimed.
- **Shared fix:** Refreshed text-gen Transformers.js wrapper for v4 result shape, opt-in WebGPU/dtype/immutable revision/progress callback and `loadForDevice` without fabricated runtime context. 83 focused text tests and full `pnpm verify` passed.
- **Follow-up:** Downstream Creator's Insight Engine switches its experimental Lil Local browser provider to this supported ONNX WebGPU lane; qualify its app-level generation and cache use separately before recommending it as a production streamer model. See `docs/verification/2026-10-19-qwen3-webgpu-transformers-qualification.md`.

## 2026-10-20 — Transformers.js text lifecycle hardening

- Scoped to `packages/text-gen`: six test-first lifecycle regressions (five RED on original) now cover single-flight initialization, device mismatch, idempotent disposal, late completion after dispose, in-flight generation cleanup, and retry after failed load. Wrapper calls the owned generator's optional `dispose()`, waits for in-flight operations, and prevents post-shutdown revival.
- `@lil-local/text-gen` tests **91/91** and typecheck PASS; full `pnpm verify` reached playground build but **failed** on `onnxruntime-web/webgpu` in separate uncommitted Bonsai2 files. Those files were preserved, not rewritten to make the gate green.
- No real-model WebGPU disposal/VRAM benchmark was run. Evidence and commands: `docs/verification/2026-10-20-transformers-text-lifecycle.md`. This is a local-only checkpoint; downstream consumers unchanged.

## 2026-10-20 — real Transformers.js WebGPU disposal acceptance (follow-up)

- Real Chrome/Chromium on NVIDIA RTX 2070 SUPER (Turing), cached Qwen3 0.6B ONNX q4 (pinned revision), **actual committed Lil Local wrapper** via temporary Vite module server; two concurrent load calls, one real generation, two idempotent dispose calls: success with no page errors. GPU usage in MiB was 4342 before model load, 5904 following generation, 4685 immediately after dispose, and 4325 after 1.4 seconds **with the browser still open**; browser close later yielded 4268. These are system-wide GPU samples, not per-process/per-buffer accounting. Output text included unfinished thinking, not an instruction-following acceptance.
- Both test Chromium sessions and temporary Vite server were terminated; no new model download or repository app changes. Evidence: docs/verification/2026-10-20-transformers-webgpu-disposal-smoke.md.
- Full pnpm verify is **still blocked** by separate uncommitted Bonsai2 onnxruntime-web/webgpu resolution. Current other-app work remains untouched.

## 2026-10-20 — Bonsai2 playground WebGPU import resolved

- Reproduced real Vite build failure for `onnxruntime-web/webgpu` in concurrent uncommitted Bonsai2 candidate. Context7 official ONNX Runtime docs confirmed import syntax; root cause was missing *direct playground dependency*.
- Added exact existing workspace `onnxruntime-web@1.30.0` to playground manifest and lockfile using offline pnpm. Full `pnpm verify`, playground build and typecheck **PASS**. No other Bonsai2 source/UI/probe edits.
- This proves production **bundling**, not shader-f16 availability, real model execution or text generation. Evidence: `docs/verification/2026-10-20-bonsai2-onnx-webgpu-package-boundary.md`.

## 2026-10-21 — Bonsai 2B candidate: source and runtime evidence

- Preserved the earlier uncommitted App / candidate panel / ONNX probe,
  then replaced misleading readiness with a verified graph audit:
  1,054,038-byte pinned decoder SHA-256 passed, 18 Scan nodes recursively
  observed. ORT WebGPU cannot execute this prefill path efficiently.
- Native fallback: built explicit loopback-only PrismML server client with
  model-ID gating and bounded prompts. UI refuses fake ONNX generation,
  explains what works, and requires a real server to expose generation.
- Isolated pinned GGUF and official Prism native release downloaded and
  checksum-verified. First **real CPU-only smoke aborted on model load**
  with GGML padded-1D scale assertion before text generation (no CUDA
  inference). This is a genuine unresolved runtime failure; see
  `docs/verification/2026-10-21-bonsai2-candidate-qualification.md`.
- Model/source artifacts kept out of Git and outside active system paths.
  No Veyax, RikkaHub, GPU-driver, ComfyUI, or sibling Bonsai Image changes.
- Pending: resolve native 2B loader compatibility (or rewrite and validate
  ONNX Scan prefill) before promotion to a usable Lil Local model. Do not
  infer generation from source-level or mocked tests.

- **UI follow-up:** Earlier click inspected the model card rather than using
  its separate Open corner action. Added an explicit Open action to the
  selected pipeline's details and corrected its "download and load" hint.
  Real Chromium (GPU-disabled) used this action to reach Bonsai's panel:
  18-Scan warning visible, local-server connect visible, no Generate control
  before model identity, no external requests or page errors.
