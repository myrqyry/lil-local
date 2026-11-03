# 2026-10-26 — Phonon ONNX real browser speech qualification

## Objective

Convert the existing local `packages/phonon` silence scaffold into a usable
Lil Local **reusable browser substrate**. Preserve unrelated edits in the
original checkout. No server, phone integration, model-store migration, or
automatic background download.

## Observed facts

- Original `run()` returned a fabricated 1,920-sample silent frame, with
  `flow_lm.run()` and `mimi_decoder.run()` commented out. A new regression
  failed because the stub reported fake success instead of missing outputs.
- Upstream immutable HF revision:
  `4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4`.
  ONNX graphs were inspected directly; true inputs include `text_ids`,
  `voice_latents`, `latent`, `noise`, conditioning tensors, recurrent
  `past_key_values.*`; outputs include `latent_out`, `eos_logit`,
  `present.*`. Mimi emits `audio`, `present_offset`, key/value and
  conv-cache tensors. Quantized graphs have separate `.onnx_data` files.
- Implemented real ONNX Runtime Web model load/infer/decoder loops, voice
  asset retrieval, audio PCM streaming callback, sample validation, graph
  ABI checks, recurrent cache/state handoff, release, and explicit backend
  selection. Published first-party `createPhononSubstrate` for
  `BrowserSubstrateRunner`; does not require a LiteRT instance.
- Real Chromium run: Q4F16 loaded but failed on `shader-f16`; Q4 variant
  then loaded both graphs and generated nonzero PCM. Browser result:
  105,600 samples at 24,000 Hz, 4.4 seconds, peak 0.294809, RMS 0.052219.
  WAV checked with ffprobe: pcm_s16le mono 24kHz, 4.4s; located on target
  machine at `/tmp/lil-local-phonon-browser.wav`. A browser console
  warning indicates some nodes were assigned outside the preferred provider.
- Target qualification is *numerically real audio*, not yet a human-confirmed
  intelligible utterance. No full GPU placement proof or long-form benchmark.

## Verification and known blockers

- Phonon focused tests and package TypeScript check: **pass**.
- Separate `pnpm test:phonon-external`: isolated pnpm pack + install +
  TypeScript + Vite build: **pass**.
- `pnpm test:boundaries`: **18 passing**.
- Broader `pnpm test:compatibility`: **fails** with TS2769 typed-array
  errors in unchanged `image-embedding/src/pipeline.ts` and
  `qwen3-tts/src/{codec,mtp,talker}.ts`. The Phonon-specific packed
  consumer passes; this is not equivalent to full `pnpm verify` success.
- Defaults still fetch graph and tokenizer from pinned Hugging Face URLs on
  first load. Voice assets come from `RuntimeContext.assets`.
  Complete offline/self-hosted routing requires a distinct tested change.

## Isolation

Branch `aster/phonon-live-onnx`, worktree
`/home/myrqyry/loam/.worktrees/lil-local-phonon-live`, based on
`bc3e04d`. The original `/home/myrqyry/loam/lil-tart` retains its
independent uncommitted Phonon scaffold and other work. No merge or push.
