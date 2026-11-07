# MediaPipe Phonon TTS — real Lil Local browser runtime qualification

Date: 2026-11-06. Branch: `aster/mediapipe-tts-qualification`, based on local `866edb8`.

## Scope

Run the existing exported `createMediaPipeTtsEngine()` and `validateTtsResult()` from `@lil-local/audio-mediapipe` in **actual headless Chromium**, backed by the actual Phonon 31M `.litertlm` and pinned MediaPipe Tasks Audio **1.2.2** WASM. This is **not** a standalone SDK demo, simulated waveform, mocked SDK, or compiled-only check. The reproduction script imports package source and serves the pre-existing model from a local-only temporary Vite middleware; it fetches no other model weights.

Official model: `litert-community/phonon-31m-en`, `phonon_int8.litertlm`, **87,205,888 bytes**, SHA-256 **b33a9a0d65acec36762db38a2bff6d899f4aab34b196bb56d89fab5665eabebe**. Locally hashed the previously cached file, and independently compared exact byte size and SHA-256 against Hugging Face's current published LFS metadata on 2026-11-06. Cached model remains outside the repository.

## Reproduce

```bash
cd ~/loam/lil-tart
LIL_LOCAL_TTS_MODEL=/absolute/path/to/phonon_int8.litertlm \
  pnpm test:tts:browser
```

Requirements: system `/usr/bin/chromium`, Playwright, pinned MediaPipe package installed and supplied matching local model. Model download is never automatic. A temporary loopback-only Vite test server and browser are closed in `finally`.

## Observed

- Real CPU/WASM engine initialization, `listVoices()`, and unary `synthesize()`: PASS. Text `Hello from Lil Local. This is real on device speech.`
- Mono PCM at **24,000 Hz**: **109,440 samples / 4.56 s**, amplitude peak **0.5149**, RMS **0.07156**, **10 word timestamps**. Unary synthesis **3.26 s** on this run; environment-specific, not a benchmark.
- Real `synthesizeStream()` on two clauses returned **19 ordered chunks**, **113,280 samples**, marked final: PASS.
- Switched to `claire` in the **same engine** for French text, got **72,960 samples**, peak **0.44115**: PASS (non-silent output, not a pronunciation-quality judgment).
- No browser `pageerror` events. Engine `close()` invoked in `finally`; did not independently measure released WASM memory.
- `pnpm verify`: **exit 0** after adding this isolated browser qualification harness.

## Discrepancy worth retaining

Google's current Text-to-Speech page calls Phonon 31M `en-US`-only, while actual pinned bundle `listVoices()` reported `freya/en-US`, `claire/fr-FR`, `lukas/de-DE`, `sofia/es-ES`, `lucas/pt-BR`. French-labeled voice emitted non-silent output for French text. This does **not** establish accurate multilingual phonemization or overall quality, and the documentation/runtime disagreement should not be silently resolved.

## Not established

- Human listening quality, multilingual intelligibility, real headphone playback or accessibility UX.
- Interrupt cancellation during *active* inference, incremental LLM token pushes, sustained long-form streaming or concurrent workloads.
- Web Worker isolation, on-phone latency, other browsers, physical WebGPU acceleration, memory release/long-session resilience.
- Kokoro/PocketTTS bundles or Audio8. Their readiness is unchanged.

No production UI, automatic downloads, cloud fallback, other applications, deployment, or push.
