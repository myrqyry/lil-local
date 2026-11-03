# Nemotron 3.5 ASR — browser ONNX qualification (2026-11-02)

## Why this runtime

Lil Local is a browser app; the GGUF/transcribe.cpp JNI adapter verified separately on Pixel 10 Pro cannot be transplanted into its LiteRT/ONNX browser worker. The browser route uses the INT4 ONNX conversion and the MIT-licensed reference implementation from [People-11/Nemotron_Web](https://github.com/People-11/Nemotron_Web), pinned upstream source `5b37ff1d9fb021db6d9db2ec3a4768e5ee6310e0`.

- Model `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`, immutable revision `8364d9e2dd9da23789b480bdbba9e423717e42ee`.
- Graphs: encoder, decoder and joint ArgMax. Joint optimized graph copied from the upstream source and shipped at `/nemotron-web/joint-argmax.onnx`. Its original parity evidence belongs to the upstream implementation, not a newly repeated local numerical parity test.
- Source license: MIT, see `apps/playground/src/substrates/nemotron-web/LICENSE.upstream.txt`.
- NVIDIA weight license metadata is inconsistent across repositories. Retained upstream third-party and NVIDIA notices under `apps/playground/public/nemotron-web/`; the app's Apache-2.0 license does **not** relicense the model.
- Model files are fetched directly from pinned Hugging Face URLs only when the user selects **Load ONNX model**. The copied browser adapter has no new service worker, avoiding conflicts with Lil Local's existing origin/cache namespace. Browser HTTP cache behavior is not an OPFS persistence guarantee.
- All inference is local to the browser; no API/cloud speech fallback. User selects `auto` (WebGPU then WASM on session-init failure), explicit `webgpu` (fail closed), or `wasm`.
- File input uses WebAudio decode, mono 16 kHz resample, original FFT/mel pipeline and worker RNNT decoding. Microphone uses `getUserMedia`, 16 kHz AudioWorklet PCM, ordered worker feeds, explicit stop/flush/finalize. **Real microphone capture was not exercised in this verification**.
- Existing LiteRT model runner and audio adapters were not modified. This is a separate opt-in pipeline in the playground.

## Verification

- Focused `nemotronWeb.test.ts` TDD: initial missing implementation RED, then 3 passing contract tests for revision/backend choice/input validation.
- `pnpm --filter playground typecheck`: exit 0.
- Full `pnpm verify`: exit 0, including typecheck, tests, boundaries, packed external consumer, conversion assets, qualification suite and production build. Build emits existing large chunk advisory and LiteRT static/dynamic import notice.
- Production-preview Chromium (`/usr/bin/chromium`, headless, local `127.0.0.1:4387`) **before model load**: visible model list entry, workspace heading, Load, file, microphone controls; 0 model-weight requests, 0 page errors.
- Production-preview Chromium actual ONNX inference: clicked **Load**, observed phases `loading encoder`, `loading decoder`, `loading joint`, `Nemotron loaded`; an English WAV synthesized with `espeak-ng -v en-us -s 135` saying “The little local speech recognition test is running in this browser.” was submitted to **Transcribe file**. The page returned **“Speech recognition test is running in this browser”**, status **File transcription complete**, and **zero page exceptions**. Total test process ~70 s includes model transfer/session setup, not an inference benchmark. ORT emitted nonfatal warnings for expected CPU-assigned shape nodes. No claim of complete word-accurate transcript, WebGPU-only operator placement, or measured hardware performance.

## Known boundaries

- Only one English synthesized file was actually transcribed locally in this browser integration. Spanish and microphone streaming still need direct acceptance runs; model language IDs exposed here are the upstream browser mapping (auto/English/Chinese/Japanese/Korean).
- Browser model file storage is not yet integrated into Lil Local's OPFS download manager. First load requires network, and HTTP cache eviction may require another download (~800 MB). Do not promise persistent offline reload.
- Large model and ORT sessions consume substantial browser memory. Repeated load/release, long microphone sessions, capture backpressure, audio flush cancellation, performance and thermal impact remain unqualified.
- No Vercel deployment, push, RikkaHub change, or phone app change was performed as part of this browser port.
