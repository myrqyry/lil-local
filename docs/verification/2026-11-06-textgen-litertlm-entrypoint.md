# Narrow browser LiteRT-LM package entrypoint

**Date:** 2026-11-06

**Trigger:** Consuming `@lil-local/text-gen` from podQast's existing Vite 6 Web Worker caused the browser bundler to load the generic Transformers.js 4 pipeline as well. Production Vite reported `onnxruntime-web/dist/ort.all.min.mjs/webgpu` as a nonexistent path. PodQast also has an established independent Transformers.js 3 Qwen/Janus path; pulling in the full text-gen package entrypoint needlessly mixed those ecosystems.

**Change:** `@lil-local/text-gen/litertlm` is a narrow public subpath exporting the LiteRT-LM pipeline, model manifest resolver, and their public input/config types. Existing root exports unchanged. The subpath doesn't itself import the Transformers.js pipeline.

**RED:** External packed consumer imported the missing subpath and `pnpm test:compatibility` failed with `TS2307: Cannot find module '@lil-local/text-gen/litertlm'`.

**GREEN:** After adding a dedicated `src/litertlm.ts` entrypoint and package export, `pnpm --filter @lil-local/text-gen typecheck` and `pnpm test:compatibility` exited 0, including all 24 external packed packages.

**Boundary:** This verifies public package resolution and compilation, not live LiteRT inference, end-to-end podcast output, or that *all* Vite versions can resolve every downstream dependency. PodQast must use this subpath and run its own production build and runtime tests. The full `@lil-local/text-gen` root entrypoint still exports the generic Transformers pipeline for consumers that need it.

**No auto download, cloud fallback or push.**
