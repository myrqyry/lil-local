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
