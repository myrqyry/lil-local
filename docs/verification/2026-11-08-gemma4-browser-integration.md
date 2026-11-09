# Gemma 4 E2B/E4B browser integration — 2026-11-08

## Correction: full E2B browser image and audio are verified

The November 7 PodQast real-browser tests had **already established** that the complete Gemma 4 E2B checkpoint, `gemma-4-E2B-it.litertlm` (2,588,147,712 bytes), performs browser image inference with GPU vision and WAV speech recognition with the CPU audio encoder through Lil Local's shared LiteRT-LM 0.19 worker. The smaller `-web` and `-gpu` exports are text-only because they lack these encoder submodels. Treating their failures and older documentation as proof of a browser API limitation was incorrect.

**Independent full-model acceptance in Lil Local's own screen:** using the complete checkpoint at revision `b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1` (SHA-256 `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`), production-built Chromium loaded the shared worker with `visionBackend: gpu`, `audioBackend: cpu`, `maxNumImages: 8`. Actual text answer: **Paris**; native second-turn memory: **turquoise railway mushroom**; real PNG image answer: **Yellow**; spoken 16 kHz mono 16-bit PCM WAV answer: **A secret word is banana.** New chat and unload passed. Full CLI exit status: **0**. No weights were bundled, and no cloud inference was invoked.

The normal Gemma screen now offers **Full E2B (text, image, WAV audio)** and retains **optimized E2B/E4B (text only)**. Audio is currently limited to canonical WAV (up to 8 MB); MP3/WebM resampling and microphone recording are not part of this browser UI. Model file selection checks name/size; the real-model verification harness also checks full SHA-256. This is evidence for Chromium on Linux, not an Android or universal-device qualification.

Reproduce with the locally cached complete model:

```bash
LIL_LOCAL_GEMMA_VARIANT=full \
LIL_LOCAL_GEMMA_WEB_MODEL=/path/to/gemma-4-E2B-it.litertlm \
  pnpm test:gemma4:browser
```

The older negative findings below refer **only to the smaller optimized web checkpoints** and must not be used to discredit the successful full E2B browser result.

## Outcome

The five archived Gemma worker/vision/audio development commits have been recovered into Lil Local's existing master checkout, without creating another worktree or active branch. The existing @lil-local/text-gen LiteRT-LM 0.19.0 worker owns loading, streaming, cancellation, and cleanup. A dedicated **Gemma 4 · Browser Inference** card is available in the normal playground model list.

**Both Gemma 4 E2B and E4B have passed real WebGPU browser text inference using the pinned official web-compatible files.** The complete full E2B LiteRT-LM file is **also browser-compatible**, as independently established by the real image/audio proof above; the optimized web exports remain distinct text-only artifacts.

To use: run pnpm --filter playground dev, open the Gemma 4 card, select E2B/E4B, select a matching -web.litertlm file or explicitly request the pinned Hugging Face download, click Load, enter a text prompt, Generate, and Unload/stop.

The model files are never bundled into the repo or automatically fetched. The real tests used an existing AI-Storage drive rather than the nearly full Linux system partition. The UI checks a local file's name and size, but does not cryptographically hash a selected 2–3 GB file; the reproducible browser smoke does hash the full file.

## Historical optimized -web model text execution

| Evidence | E2B WebGPU | E4B WebGPU |
| --- | --- | --- |
| File | gemma-4-E2B-it-web.litertlm | gemma-4-E4B-it-web.litertlm |
| HF repo | litert-community/gemma-4-E2B-it-litert-lm | litert-community/gemma-4-E4B-it-litert-lm |
| Immutable revision | b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1 | 2eee7ac325f20eb8c9ac1d0e972f7c84663062da |
| Bytes | 2,008,432,640 | 2,969,059,328 |
| SHA-256 | 3a08e8d94e23b814ae5414469c370c503813949acb8ceaa17e4ebf8a35af35b5 | 3904d826d5dddd25ea173e85204caec09e68ba038116e9b992b69cbdc94f57a0 |
| Full local SHA-256 | PASS | PASS |
| Native model load through production UI | 3,826 ms | 4,794 ms |
| Actual output | The capital of France is Paris. | The capital of France is Paris. |
| Real browser script | PASS / exit 0 | PASS / exit 0 |

The models were loaded through **production-built** Lil Local in a headless Chromium instance on Linux with a WebGPU adapter, using the official @litert-lm/core 0.19.0 SDK and the actual classic worker. The test checks: no model auto-load, pinned file SHA/byte count, self-hosted LiteRT-LM WASM availability (HTTP 200), real model initialization, streamed native text, output semantic correctness, and explicit release via Unload/stop. The model and prompts are not sent to a remote inference API. The physical GPU identity and actual native GPU placement were not independently established from WebGPU adapter details.

This is **limited browser text qualification** on one environment, not a claim about mobile WebGPU, other browsers, 100% model accuracy, or native Android image/audio inference. Both web manifests explicitly record pass/limited with the relevant browser environment. Remote HF URI streaming was not exercised in these sessions; only local official files were used.

## Native chat continuity and thinking verification

The supplied official LiteRT-LM JavaScript reference documents persistent
conversations and the getTokenCount, getBenchmarkInfo and clone APIs. Inspection
of the installed 0.19.0 SDK confirms those capabilities. The previous Gemma
screen created and deleted a fresh native conversation on every turn. This
revision corrects that specific missing behavior.

The shared text-gen worker now supports an optional sessionId and retains
at most one native Conversation per loaded engine. Consecutive prompts with
the same ID reuse the SDK conversation and native KV state, without replaying
history. Changing chat identity starts a fresh conversation; unloading deletes
it. Cancellation or generation errors invalidate the retained conversation,
and incompatible per-chat settings are rejected rather than silently ignored.
Existing one-shot consumers retain their earlier behavior when sessionId
is omitted.

The playground adds New chat, an in-memory transcript, an opt-in thinking
toggle, an expandable reasoning channel and SDK token counts. Thinking mode
uses the documented preface.extra_context.enable_thinking setting; toggling
it starts a new native session.

**Actual production-browser E2B and E4B tests:** each first prompt supplied
the arbitrary secret code "turquoise railway mushroom" and asked for
France's capital. Both answered "Paris". Each second prompt asked what code
was supplied in the previous turn, without including the code itself.
Both returned "turquoise railway mushroom". The browser also cleared
visible history on New chat and unloaded the engine. Both files were
independently SHA-256 verified before the test.

**Actual E2B thinking test:** after enabling thinking for a fresh conversation,
the native model answered "4" for 2 + 2 and exposed a reasoning channel.
E4B thinking was not separately runtime-qualified.

The SDK returned native session token counts of 35 then 69 on the two-turn
test. Its browser benchmark prefill/decode rates were both zero, so the UI
displays them as unavailable instead of presenting misleading throughput.

Reproduction: run the existing test:gemma4:browser command with the pinned
E2B/E4B files documented below. To exercise the optional real E2B thinking
test, set LIL_LOCAL_GEMMA_THINKING_PROBE=1. Focused worker regression tests
cover reuse, cleanup, settings mismatch, thinking config, cancellation and
one-shot compatibility.

The source docs also advertise Conversation.clone() and getHistory().
These remain SDK capabilities, not claimed user-facing clone/history-export
features; the conversation is memory-only and does not persist on reload.
The image/audio failures below remain unaffected.

## Historical optimized -web export image/audio failures (not full E2B)

Actual E2B WebGPU model probes (not mocked tests) also attempted:
- **Image:** loaded with visionBackend=gpu and maxNumImages=1, then supplied a generated 192x192 red PNG. The native SDK failed: **Vision executor should not be null, please TryLoadingVisionExecutor() first.**
- **Audio:** loaded with audioBackend=gpu, then supplied a locally synthesized spoken WAV. The WASM runtime emitted **Aborted(). Build with -sASSERTIONS for more info.** No usable transcript/text result was observed.

The JavaScript Web API and these -web.litertlm files have not demonstrated working image/audio executors, even though native Gemma 4 models support multimodal tasks. Therefore the **broken image/audio choices were removed from the browser UI**. The recovered typed worker protocol and its source remain available for a future compatible native execution path, but they are not presented as qualified features. The worker now **rejects known E2B/E4B -web checkpoint loads with image/audio options before WASM initialization**, instead of letting the native model abort; this is guarded by a RED→GREEN regression test. There is no silent cloud fallback or substitution with another model.

## Reproduce with your own pinned files

From the Lil Local repo root, after pnpm install:

    pnpm --filter @lil-local/text-gen exec vitest run src/worker/litertlm.worker.test.ts src/gemma4-web.test.ts
    pnpm --filter playground exec vitest run src/App.test.tsx src/components/Gemma4Panel.test.tsx
    pnpm verify

    LIL_LOCAL_GEMMA_WEB_MODEL=/path/to/gemma-4-E2B-it-web.litertlm pnpm test:gemma4:browser
    LIL_LOCAL_GEMMA_VARIANT=e4b LIL_LOCAL_GEMMA_WEB_MODEL=/path/to/gemma-4-E4B-it-web.litertlm pnpm test:gemma4:browser

The browser model test verifies the full SHA-256 every time. For non-model UI/GPU/WASM probing:

    LIL_LOCAL_GEMMA_PROBE_ONLY=1 pnpm test:gemma4:browser

Generated LiteRT-LM WASM files come from the exact installed npm package and are hashed and copied by scripts/prepare-litertlm-wasm.mjs into a gitignored browser vendor directory on predev/prebuild. No runtime CDN dependence, model bundling or implicit model download is introduced.

## Historical integration

The original archived candidate line was recovered from the published Git tag archive/lil-local/gemma4-audio-2026-11-08, by applying its test delta to master before the worker implementation. Before source integration 4 of 16 worker regression tests failed; after integration all 16 passed. The three original archive tags remain provenance only. There are no new branches or worktrees awaiting a merge.

Source:
- packages/text-gen/src/worker/ and packages/text-gen/src/litertlm.ts — recovered original worker, SDK WASM initialization and typed protocol.
- packages/text-gen/src/manifest.ts and src/gemma4-web.ts — separate WebGPU artifacts, pinned revisions and browser text qualification metadata.
- apps/playground/src/components/Gemma4Panel.tsx — actual model input, generation and unload UI.
- scripts/prepare-litertlm-wasm.mjs — reproducible local 0.19.0 runtime assets.
- scripts/gemma4-browser-smoke.mjs — real E2B/E4B browser qualification.

Only the smaller optimized -web model file has negative image/audio findings. The **complete E2B model has positively qualified browser image/audio inference** as documented above. A future real browser SDK executor or native consumer must be tested with a real model response before either is enabled.
