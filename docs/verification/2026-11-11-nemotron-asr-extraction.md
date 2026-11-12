# Nemotron ASR packaging and worker boundary — 2026-11-11

## Ownership and verified goal

Lil Local previously bundled a real Nemotron 3.5 ASR ONNX implementation only
inside `apps/playground/src/substrates/nemotron-web`. A public package was
needed before Veyax or other apps could reuse the proven inference path.

Moved the original MIT-adapted engine, worker, client, audio frontend and
associated model-contract tests into `@lil-local/nemotron-asr`, preserving
model revision `8364d9e2dd9da23789b480bdbba9e423717e42ee`, ONNX Runtime
Web 1.30.0, backend policy and model semantics. The playground and its
Parakeet/YAMNet shared audio decode consumers now import the public package.
The small `joint-argmax.onnx` remains served by the playground at its
historical `/nemotron-web/joint-argmax.onnx` URL; external apps must host it
and pass their URL through the optional asset config. No model weights were
copied or downloaded, and the license notices remain separate from source.

## Regression and platform checks

- RED: new external public module could not be imported until extraction.
- GREEN: model/backend contract tests plus worker-client tests; PCM transfer
  now copies caller-owned samples instead of detaching their buffers.
- `pnpm --filter @lil-local/nemotron-asr test`: 6/6; package and playground
  typechecking passed.
- Independently packed `pnpm test:compatibility`: **exit 0**, including the
  new package install, import, contract tests, TypeScript and Vite build.
- On localhost Vite port 5271, real headless Linux Chromium dynamically
  imported the package, instantiated the real Worker and awaited `dispose()`.
  Reported worker URL: `/@fs/.../packages/nemotron-asr/src/nemotron.worker.ts`.
  **No page exceptions or external requests**, no model initialization.
- Full root `pnpm verify`: **exit 0**, covering workspace typecheck/tests, WebGPU kernel-selection checks, dependency boundaries, independently packed external compatibility, conversion-asset checks, runtime qualification, and production builds. The resulting playground bundle included `dist/assets/nemotron.worker-lQjDQy82.js` (124.93 kB). Existing chunk-size/dynamic-import advisories remain nonfatal.

## Limits and continuation

The earlier `2026-11-02-nemotron-browser-asr.md` record did execute real
English transcription, but **before this code relocation**. Cold-worker and
pack/build checks do not prove that end-to-end ONNX inference still works
when the relocation is exercised with the model. Model weights total around
800 MB; the main Loam filesystem is ~100% used with ~2.5 GB free, so no
repeat download is initiated without a deliberate resource decision.

The original Veyax watch-comment goal remains open. Veyax pins a historical
Lil Local commit missing these inference packages, and YouTube capture currently
has `audio: false`. Downstream integration requires coherent same-SHA package
repinning, explicit tab audio capture with permission, ASR session timestamp
mapping, then audible commentary tests. Do not duplicate the Nemotron engine
in Veyax, silently use cloud speech, or claim a live YouTube acceptance test.
