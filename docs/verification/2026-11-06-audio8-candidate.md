# Audio8 Realistic Air: candidate asset contract (2026-11-06)

## Source / scope

Pinned conversion: https://huggingface.co/litert-community/Audio8-TTS-Realistic-0.6b/tree/752cd650bf5f649f2c0139ea02ac5627e9e60a5a

Local branch `aster/audio8-litert-candidate`, created from clean `aster/mediapipe-language-detection` at `09f1186`. No model downloads or browser inference attempted.

New `packages/audio8-tts` provides a public manifest for three asset profiles and preserves upstream SHA-256, size, variant-specific dependencies, revision and honest unqualified verification status. It does **not** expose an inference pipeline.

## Evidence

- RED: `pnpm exec vitest run packages/audio8-tts/src/manifest.test.ts` failed importing the missing manifest.
- GREEN: same command, 3/3; `pnpm --filter @lil-local/audio8-tts typecheck` exit 0.
- `pnpm install --offline --ignore-scripts` exit 0; lockfile updated for the new package.
- `pnpm verify` exit **0**; includes package compatibility, boundaries, full tests, qualification and build. Browser tests of Audio8 specifically were not run because no browser inference implementation exists.
- Independent Python script fetched pinned upstream `SHA256SUMS` and compared all 10 local sha256 fields by path: **10/10 match**. This verifies published metadata transcribed correctly, **not** local payload bytes.

## Explicit limits

Publisher validates Android LiteRT 2.2; Lil Local's web LiteRT.js is a separate substrate and may reject graph operators, signatures or stateful KV cache patterns. No working synthesis, browser backend performance, voice cloning, GPU accuracy or output parity is claimed. Resident memory is a conservative planning estimate, not measured browser usage.

Next falsification: try the slow/fast graph's named signatures and codec on the actual browser runtime; only then port prompt, sampler, generation, and audio output end to end.
