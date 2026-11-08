<!-- meristem-template:v1 -->
# Project

Authority note: this file is a Meristem index, not a replacement for the
repository's own docs. Where a section below points at an established
authority, that authority wins and this file should be updated to match it.

## Purpose

Lil Local is a local-first browser inference workspace with two faces that are
one project: the interactive app in `apps/playground` (a model launcher and
runtime qualification lab) and the reusable `@lil-local/*` packages
that app is built from.

The distinctive premise (from `README.md`): **prove a capability once, package
it once, and consume it everywhere**, instead of rebuilding browser inference
plumbing in every app. The intended workflow is deliberately boring — browse
models without triggering a download, explicitly download and load the model
you want, observe which backend actually resolved, run preflight and real
inference, then capture enough evidence to reuse that working path elsewhere.

Coverage spans text generation, retrieval, speech, audio, vision, OCR, image
embeddings, and video classification.

## Current mainline and branch boundaries (2026-11-07)

- Local `master` was **fast-forwarded** to the verified MediaPipe / Audio8
  candidate / SlimSAM / Live Vision development tip (`5ce3865`), preserving
  19 original commits; a later documentation-only checkpoint (`3a81dca`)
  records the consolidation. The 15 redundant local feature refs were safely
  retired after ancestry checks, leaving `master` plus the three distinct
  Gemma worker/vision/audio branch pointers. The branch ledger is
  `docs/verification/2026-11-07-mainline-reconciliation.md`.
- Live Vision reuses its existing camera/still-image workbench. Nine opt-in
  MediaPipe frame tasks (including the Selfie foreground confidence model) and
  Depth Anything V2 Small form the ten-option selector. Browser-qualified
  Selfie and Q8/WASM depth are separate from unverified FP16/WebGPU or Android
  performance; no real-time claim is implied.
- Gemma's separate five-commit development line has passed focused text-gen
  typechecks and unit tests, **not** real Gemma vision/audio model inference.
  It is not part of `master`; its worktree stays isolated.
- GitHub synchronization status must be read from live refs; the reconciliation
  receipt records the local-only state **at the time of that audit**, not a
  permanent claim that published `origin/master` is behind.

## Invariants

- `pnpm verify` is the authoritative pre-merge gate. Boundary and qualification
  suites are not optional extras.
- The shared packages use `@lil-local/*` and the canonical repo is
  `myrqyry/lil-local`. The scope was deliberately renamed alongside branding;
  downstream consumers must migrate all pins together.
- Packages ship TypeScript source: no per-package build step,
  `exports: ".": "./src/index.ts"`, all `private: true` and `type: module`.
- A package is only part of the supported downstream surface when
  `pnpm test:compatibility` can pack, install, type-check, and build it from its
  public entrypoint. Workspace-only dependencies must not leak as `workspace:*`
  into packed metadata.
- Consumers pin one Lil Local Git SHA per application and resolve every Lil Local
  package from that same SHA. Do not mix revisions; peer ranges are compatibility
  information, not permission to mix.
- Product concerns (episodes, OBS, UI, orchestration, conversation semantics)
  belong in consuming apps, not in shared packages.
- A successful app session does not silently promote repository verification
  metadata. Runtime evidence and durable verification level are separate.

## Architecture and boundaries

Dependency flow: `inference-core → runtime-litert → model packages → playground`.

- `packages/inference-core` — contracts, assets, receipts, validation. No model
  dependencies; must stay independent of the runtime.
- `packages/runtime-litert` — managed LiteRT runtime, backend selection, caching,
  preflight, telemetry.
- Model packages — `text-gen`, `retrieval`, `encoder`, `image-embedding`,
  `kokoro`, `qwen3-tts`, `video-classification`, `depth-estimation`
  (including pinned Depth Anything V2 q8/WASM and unqualified FP16/WebGPU),
  `image-segmentation` (including optional SlimSAM-77 q8 point selection),
  the opt-in `vision-mediapipe` nine MediaPipe frame tasks (including Selfie
  foreground confidence, BlazeFace/face/gesture/pose/object/classifier/holistic/
  semantic) plus separate MagicTouch still-image selection,
  `text-mediapipe` local language detection (MediaPipe Tasks Text 1.2.2),
  `audio-mediapipe` sound/ASR/TTS,
  and `retrieval-mediapipe` universal embedding/index substrates,
  and the custom-WebGPU `pix2pix-webgpu` image-translation candidate.
- `packages/browser-cache` — browser-side model/tensor cache.
- `apps/playground` — the consumer app (React + Vite + Tailwind).
- `examples/` — minimal consumer examples, deliberately small.
- `tests/` — cross-package boundary tests; `tests/runtime-qualification/` is a
  Node-environment runtime qualification suite with its own vitest config.

Where authority lives (do not duplicate these):

- Agent rules and nested routing: `AGENTS.md` plus the six nested
  `AGENTS.md` files for `runtime-litert`, `qwen3-tts`, `video-classification`,
  `inference-core`, `tests/runtime-qualification`, and `apps/playground`.
- Downstream revision policy and supported surface: `docs/package-revision-policy.md`.
- Durable verification records: `docs/verification/`.
- Change history: `CHANGELOG.md` (Keep a Changelog, SemVer, active `[Unreleased]`).
- `docs/superpowers/` is a git-ignored local working lane for plans/specs, not
  shared project knowledge.

## Known-good workflows

| Command | Meaning |
| --- | --- |
| `pnpm verify` | CI gate: typecheck + test + boundaries + compatibility + conversion assets + qualification + build |
| `pnpm test:boundaries` | Package dependency rules (examples → public entrypoints, inference-core independence, playground → runtime-litert) |
| `pnpm test:compatibility` | Pack and consume the supported external surface |
| `pnpm test:conversion-assets` | Validate pinned conversion recipes and candidate-evidence boundaries |
| `pnpm test:qualification` | Runtime qualification suite (separate vitest config, Node env) |
| `pnpm test:vision:browser` | Real-browser regression for eight preexisting MediaPipe frame tasks and camera lifecycle |
| `pnpm test:controlnet-models:browser` | Real Chromium Selfie and Depth Anything V2 Q8 inference through Live Vision |
| `pnpm test:slimsam:browser <image> <x> <y>` | Opt-in real SlimSAM point-mask browser probe |
| `pnpm --filter <pkg> test` / `typecheck` | Single-package work |
| `pnpm dev` | Playground dev server |

Environment: Node 22+, pnpm 11+.

Workflow rules in force (from the root `AGENTS.md`): stage by explicit path, never
`git add .`; conventional commit messages with a bullet body for multi-area
changes; the pre-push hook runs automatically and real failures must be fixed
rather than bypassed.

CI mirrors this: `.github/workflows/verify.yml` on push to `master` and pull
requests; `.github/workflows/runtime-qualification.yml` is manual
`workflow_dispatch` with optional case/backend inputs.

## Verification

Distinguish evidence classes rather than collapsing them:

1. Static inspection — reading source and config.
2. Build/typecheck/test evidence — `pnpm typecheck`, `pnpm test`, boundary and
   compatibility suites.
3. Runtime evidence — actual preflight and inference in a browser, backend
   resolution observed rather than assumed.
4. Manual UI verification — only for user-visible behavior.

Only claim the level actually established. A commit that once passed verify on
an earlier branch is not evidence about the current tree.

Browser observations are a separate, recorded evidence class. They live in
`docs/verification/` as durable records, because the raw results JSON under
`tests/runtime-qualification/results/` is gitignored. Run one with
`pnpm qualify -- --case <id>`. A new browser case is only worth trusting once it
has been shown to fail for the right reason.

## Resolved boundaries and open maintenance risk

- **Resolved — package subpaths:** `docs/package-revision-policy.md` explicitly
  treats exported subpaths as supported on the same terms as `.` **only when**
  `pnpm test:compatibility` resolves them through packed tarballs. The supported
  `@lil-local/retrieval/scoring` subpath is included. Do not reopen this as an
  unresolved package policy question.
- **Still open — duplicated guidance:** `AGENTS.md` and
  `docs/package-revision-policy.md` both contain workflow conventions and
  nested-agent routing. They currently agree, but have a synchronization
  maintenance cost. Choosing one authority is a separate decision; do not
  silently rewrite or remove either.
- **Still open — experimental Gemma branch:** real vision/audio inference and
  integration against the published mainline must be qualified before it can
  be called production-ready or merged. See the branch ledger above.
