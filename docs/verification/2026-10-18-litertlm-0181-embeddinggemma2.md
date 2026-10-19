# LiteRT-LM 0.18.1 / EmbeddingGemma 2 integration, October 18, 2026

## Provenance

- npm registry `@litert-lm/core@0.18.1`; installed through pnpm with
  the package exact-pinned in `packages/text-gen` and the new
  `packages/embedding-litertlm`. Root `pnpm-lock.yaml` resolves that same version.
- Existing upstream `EmbeddingEngine.create(settings)`,
  `computeEmbedding(input, {normalize, outputSize})`, and `delete()`
  are present in the package's installed **declaration files and JS implementation**.
  The source was inspected before writing the new adapter.
- Hugging Face metadata query for
  `litert-community/embeddinggemma-2-740m-litert-lm` returned revision
  `24d962e906c7d332c6428e71c9676855024569e2`, repository not gated,
  and `embeddinggemma-2-740m.litertlm` listed as an asset. That immutable
  source revision is exported, not a verified local SHA-256 or browser parity receipt.
- Official Google Web docs describe the text embedding API and task prefixes:
  https://developers.google.com/edge/litert-lm/embedding_models .

## Changes and scope

- Upgraded the existing text generation package from
  `@litert-lm/core@0.15.0` to exact `0.18.1`, preserving existing
  pipeline and worker source contracts.
- Added dedicated, opt-in `@litert-playground/embedding-litertlm` package:
  explicit caller-owned model source, WebGPU text-only embedding, streaming
  cancellable HTTP fetch, late engine release through shared substrate runner,
  explicit search prefixes, 128/256/512/768 dimension handling, finite-output
  validation, no implicit download or model execution.
- Added this package to the packed external-consumer compatibility harness.
- RikkaHub Local `litertlm-android:0.18.0` and installed mobile models
  were not changed.

## Evidence classes

1. **Static:** npm package exported types and implementation inspected; real
   upstream API confirmed.
2. **Unit/mocked integration:** new 7 EmbeddingGemma contract cases pass,
   including lazy loading, stream fetch, cancellation, late cleanup, prompt
   prefix formatting, shape and NaN/Infinity guard. Existing text-gen suite:
   74 tests pass.
3. **Packed external consumer:** new package was packed, installed, typechecked,
   and bundled through its public entrypoint by `pnpm test:compatibility`;
   successful isolated consuming-app build is not proof of model inference.
4. **Full repository gate:** `pnpm verify` **exit 0** on final source tree; package boundaries, compatibility, conversion checks, qualification test contracts and all builds included. The existing chunk-size warnings were advisory. A red test caught a semver-range manifest slip (`^0.18.1`); it was fixed to exact `0.18.1` in both text-gen manifest and lockfile.
5. **Actual model/device inference:** **NOT PERFORMED**. No EmbeddingGemma
   weights downloaded or model invocation attempted. Upstream SDK version
   change has not yet been benchmarked with a real text generator either.

## Remaining before qualification

Real WebGPU load with pinned EmbeddingGemma 2 artifact, consistent finite
embeddings at multiple output dimensions, numerical reference/parity checks,
query/document retrieval ranking checks, repeated load-dispose-cancel and GPU
memory observations on target device, and transport-abort/hardware-failure
evidence. Multimodal image/audio support remains outside the text-only adapter.
