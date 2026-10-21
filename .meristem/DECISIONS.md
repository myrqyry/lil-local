<!-- meristem-template:v1 -->
# Decisions

Record decisions only when the choice or its rationale is likely to matter later.

Existing authority: settled downstream-consumption decisions (single Git SHA per
application, no mixed revisions, no `src` imports, supported packed surface) are
already durable in `docs/package-revision-policy.md` and are not duplicated here.
Add entries below only for choices that are *not* already recorded there.

Use entries like:

## YYYY-MM-DD — Decision

**Decision:**  
**Why:**  
**Alternatives considered:**  
**Constraints / consequences:**  
**Evidence / references:**  
**Supersedes:** none

## 2026-09-30 — Keep conversion evidence separate from browser qualification

**Decision:** Android LiteRT/CompiledModel conversion results may enter Lil Tart as pinned conversion evidence and candidate-model metadata, but they do not promote browser verification. Browser qualification still requires Lil Tart's own recorded browser observation.

**Why:** Conversion success, delegate compilation, and even successful inference can diverge from numerical correctness across runtimes and devices.

**Alternatives considered:** Treat upstream Android success as generic LiteRT qualification; rejected because it would collapse distinct runtime evidence classes.

**Constraints / consequences:** Candidate models remain browser-unverified until qualified here. Applicable runtime cases should attach numeric parity evidence when a trusted reference output exists.

**Evidence / references:** `conversion/`, `tests/runtime-qualification/shared/numericParity.ts`, source snapshot `john-rocky/LiteRT-Models@460d52221c22388b6a9a8e8a44b61dde72976b4e`.

**Supersedes:** none

## 2026-10-12 — Qualify D1 Omni text decisions with a separate ONNX WebGPU substrate

**Decision:** Add an opt-in, pinned-revision D1 Omni 600M Q8 text decision package and browser laboratory panel. Keep ONNX Runtime Web separate from the shared LiteRT runtime. Do not automatically load 400+ MB of weights or repurpose Veyax's working native GGUF path. This slice supports choice and yes/no only, reports uncalibrated probabilities, and does not authorize actuation.

**Why:** The ONNX Community conversion exposes a real WebGPU-compatible six-input decision graph. Veyax already has native D1 Mira/Vela semantics, but Lil Tart had no browser execution/proof path. One model family can now be evaluated in multiple substrates with independent runtime evidence.

**Constraints / consequences:** License is LFM Open License v1.0; versioned model assets are downloaded only on click. WebGPU is explicitly requested, though ORT may execute some shape operations on CPU. Vision, voice, numerical parity and long-running stability are not qualified. Use package-compatibility and actual model inference evidence, never just a successful build.

**Evidence:** \`packages/decision-onnx/\`, \`apps/playground/src/components/D1DecisionPanel.tsx\`, \`docs/verification/2026-10-12-d1-webgpu-text.md\`; real Chromium NVIDIA/Turing Q8 text decision on 2026-10-12, single-run billing 99.56%, ~1011 ms.

## 2026-10-13 — Multi-substrate browser inference without collapsing runtimes

**Decision:** Lil Tart is a browser inference launcher, lab, and reusable
runtime qualification system, not solely a LiteRT library. Keep the established
LiteRT runner and per-model ABI intact. Introduce `BrowserSubstrateRunner` in
`inference-core` as a small, public engine-independent explicit
load/infer/dispose contract. Migrate the already existing D1 Omni ONNX Runtime
Web *panel wiring* through a real, separately implemented ONNX adapter.
Keep GGUF/llama.cpp and other candidate engines uninstalled until individually
qualified. Runtime identity, model format, task ABI, and WebGPU/WebNN/WASM
execution backend are distinct concepts.

**Boundaries:** No automatic runtime/model downloads; no engine-neutral tensor
ABI, global GPU allocator, cross-runtime caching, universal preflight,
automatic backend selection, or auto-promotion from session output to durable
qualification. D1's previously observed browser path requested WebGPU but did
not establish GPU placement for every ONNX node. This new UI bridge has only
static, mocked-integration, and build evidence until re-smoked on a real model.

**Verification:** `pnpm --filter @litert-playground/inference-core typecheck`
and tests passed (47 tests including new lifecycle cases); playground
typecheck and 169 tests passed; `pnpm verify` exited 0 on the final
runtime/consumer source tree, exercising the packed entrypoint. New tests
cover explicit lazy load, real invocation delegation through an injected
engine, single-flight inference, late-load cleanup, disposal waiting, errors,
and external package API compatibility. No new live browser inference is
claimed by these tests.

**References:** `docs/web-substrates.md`,
`packages/inference-core/src/substrate.ts`,
`apps/playground/src/substrates/d1Onnx.ts`,
`apps/playground/src/components/D1DecisionPanel.tsx`,
`docs/verification/2026-10-12-d1-webgpu-text.md`.

## 2026-10-21 — Foreground-mask package before Forma integration

**Decision:** Extract a reusable, browser-local U²-Net saliency package
(`@lil-local/image-segmentation`) from existing Lil Local adapter knowledge
without copying playground UI or extending Veyax's legacy dependency pins.
The new package owns tensor preprocessing, output shape/alpha contracts,
WASM-first runtime lifecycle, and position-preserving cropped RGBA extraction.

**Why:** Forma needs an automatic foreground suggestion that can be
corrected with its existing manual layer editor. Prior See-through GPU
depth processing wedged the machine. Start with an 88.2 MB candidate,
not a multi-gigabyte layered diffusion model; permit GPU qualification
later under a controlled workload rather than prohibiting it.

**Evidence/limits:** Model artifact metadata is source-verified and
pinned by revision/SHA. The package's unit tests and external-consumer
harness prove code/API behavior. They DO NOT prove model output or a
completed app workflow. A full inference qualification and Veyax migration
reconciliation are still required. Lil Local's original worktree contains
unrelated in-progress Bonsai files and was left unchanged; this work is
isolated in `aster/forma-image-segmentation`.

**Consequence:** Do not reintroduce `@litert-playground/*` in new Veyax
integration code and do not silently mix pinned Lil Local Git SHAs.
Model suggestions never claim to infer missing covered-up artwork or
semantic Live2D parts.

**Supersedes:** none
