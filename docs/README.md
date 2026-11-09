# Lil Local documentation

Lil Local is both the browser model lab (`apps/playground`) and the reusable
`@lil-local/*` inference packages. **A passing build is not proof that a
checkpoint runs on a particular backend.** Use the evidence for the exact
model, source, input, browser and device that were tested.

## Start here

| Need | Reference |
| --- | --- |
| Overview, setup, package inventory and everyday commands | [Project README](../README.md) |
| Supported public package APIs and single-revision Git pinning | [Package revision policy](package-revision-policy.md) |
| Substrate architecture and backend qualification boundaries | [Web substrates](web-substrates.md) |
| **Real Gemma 4 E2B/E4B WebGPU text inference and image/audio negative results** | [Gemma 4 browser qualification](verification/2026-11-08-gemma4-browser-integration.md) |
| Historical consolidation, retained commits and unmerged branches | [Mainline reconciliation](verification/2026-11-07-mainline-reconciliation.md) |
| Current work vs finished, archived or unqualified experiments | [Experiment dispositions](archive/2026-11-08-experiment-dispositions.md) |
| Historical dated implementation notes (not an active task queue) | [Active work](../ACTIVE_WORK.md) |
| Historical remote branch tips and ancestry | [Remote branch inventory](archive/2026-11-08-remote-branch-inventory.md) |
| Developer conventions and test gates | [Repository AGENTS.md](../AGENTS.md) |
| Per-model observations, inputs, metrics and reproducible evidence | [Verification records](verification/) |

## Gemma 4 browser integration

The normal model list now includes **Gemma 4 · Browser Inference** for E2B/E4B text generation through the existing LiteRT-LM worker, with locally hosted exact-version WASM and optional official SHA-pinned WebGPU model files. Both models completed actual text inference in Chromium. The native image/audio configuration options were recovered into the worker API, but the web E2B checkpoint failed real image and WAV execution, so the browser UI deliberately remains text-only. [Read the exact model/backend evidence and reproduction commands](verification/2026-11-08-gemma4-browser-integration.md).

## Live Vision and segmentation

The existing Live Vision app supports opt-in webcam or still-image input with
**nine MediaPipe frame processors**: face landmarks, gestures, body pose,
object detection, classification, holistic landmarks, DeepLab semantic
classes, BlazeFace boxes and **Selfie Segmenter person confidence**.
Its **tenth option** is Depth Anything V2 Small via the shared
`@lil-local/depth-estimation` package. Camera permissions, frame scheduling,
user interface and preview policy remain in `apps/playground`, not in the
shared package implementations.

| Capability | What was actually verified | Reference |
| --- | --- | --- |
| Existing eight MediaPipe frame processors | Real Chromium image / video inference and camera cleanup; backend distinctions in each receipt | [Vision regression](verification/2026-11-04-mediapipe-vision-substrate.md) |
| MediaPipe Selfie Segmenter | SHA-256-pinned 249,537-byte native model; 640×640 nonconstant foreground confidence mask and fake-camera VIDEO mode | [ControlNet model proof](verification/2026-11-07-controlnet-models.md) |
| Depth Anything V2 Small | Revision-pinned ONNX, Q8/WASM in Chromium; self-hosted ONNX Runtime; 640×640 nonconstant relative-depth preview | [ControlNet model proof](verification/2026-11-07-controlnet-models.md) |
| MagicTouch V2 interactive strokes | Pinned model and actual browser still-image mask changes; several seconds per CPU stroke | [MagicTouch proof](verification/2026-11-04-mediapipe-interactive-segmentation.md) |
| SlimSAM-77 | 13.15 MiB Q8 encoder/decoder and actual point-prompted masks in Chromium | [SlimSAM proof](verification/2026-11-07-slimsam77-quantized-browser.md) |

**Do not interpret** the 8.3 FPS *camera sampling cap* as delivered processing
FPS. The observed Depth Anything V2 Q8/WASM request took **2,636 ms** on the
tested headless desktop Chromium path. Actual WebGPU FP16 execution, native GPU
placement, Android performance, temporal depth stability, per-frame accuracy,
and automatic ControlNet diffusion conditioning remain unqualified. The
ControlNet Live standalone application has not been replaced or deleted.

The depth output is a normalized, relative `DepthFrame`, not metres.
The Selfie output is a float foreground probability field, not DeepLab
semantic class IDs or per-person instance masks. Grayscale is a UI preview,
not the shared raw result.

For API usage see [Vision package](../packages/vision-mediapipe/README.md)
and [Depth package](../packages/depth-estimation/README.md).

## Other recently qualified or candidate capabilities

- [Audio/YAMNet/ASR substrate](verification/2026-11-04-mediapipe-audio-substrate.md)
  and [MediaPipe TTS session](verification/2026-11-06-mediapipe-tts-consumer-session.md)
- [Language identification](verification/2026-11-04-mediapipe-language-detection.md)
- [Universal embeddings and retrieval](verification/2026-11-04-mediapipe-samples-retrieval.md)
- [Audio8 model candidate](verification/2026-11-06-audio8-candidate.md):
  asset/provenance profile, **not** a ready-to-run TTS engine
- [Phonon ONNX speech](verification/2026-11-06-mediapipe-phonon-tts-browser.md)
- [SHARP ONNX 3D adapter](verification/2026-10-26-sharp-onnx-package.md):
  package and decoding work; full predictor inference not established

## Reproduce the evidence

Run from the repository root with Node.js 22+ and pnpm 11+.

```bash
pnpm install
pnpm verify
pnpm test:vision:browser
pnpm test:controlnet-models:browser
# Optional: requires a caller-supplied real image and coordinates:
pnpm test:slimsam:browser /path/to/input.png 360 170
```

`pnpm verify` tests types, unit contracts, package boundaries,
packed external consumption, conversion assets, qualification contracts
and production bundles. It does **not** itself run all large model weights.
Browser smokes intentionally download upstream test references or model
weights; offline execution requires those assets to be supplied/cached,
with matching runtime files served locally. Inference is local after load;
an initial network download is not equivalent to remote inference.

## Branches and historical evidence

The [2026-11-07 mainline audit](verification/2026-11-07-mainline-reconciliation.md) records the original 19-commit fast-forward. **The active Lil Local branch list now contains only `master`**, locally and on GitHub, with one worktree. All 26 retired topic-branch tips are reachable via one published annotated archive tag. The three unqualified Gemma stage commits also have their own archive tags. No unverified work was merged. The [disposition and recovery ledger](archive/2026-11-08-experiment-dispositions.md) and [former branch inventory](archive/2026-11-08-remote-branch-inventory.md) preserve what was actually done.

The audit's `origin/master` ahead/behind observations are a historical
snapshot before publication. Use `git fetch origin` and
`git status -sb` to determine present remote synchronization; do not
interpret a commit being pushed as proof that CI, preview deployments or
real-device qualification succeeded.
