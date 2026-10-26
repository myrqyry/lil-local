# 2026-10-26 — OpenAI Privacy Filter browser integration

## Intent and bounds

Enable opt-in OpenAI Privacy Filter inference as reusable Lil Local substrate.
For the current Linux Mint browser environment choose the **smallest variant
that really runs**. Preserve larger model choices for other consumers. Do not
automatically edit stored conversations, scan sensitive documents, send prompts
to a service, or merge/push a branch without a reason.

## Sources / on-device facts

- Official Hugging Face model `openai/privacy-filter` at revision
  `7ffa9a043d54d1be65afb281eddf0ffbe629385b`.
- Published ONNX graph sizes from Hub metadata: Q4F16 sidecar
  809,061,992 bytes, Q4 sidecar 917,120,144 bytes. This is **not** an estimate
  of GPU-resident memory. Tokenizer JSON ~27.9 MB additional.
- Actual Chromium WebGPU adapter is available but reports
  `shader-f16 = false`. Therefore default chooses **Q4**, rather than smaller
  Q4F16. Other users can select compatible backends/precision explicitly.
- Browser model load fetched config, tokenizer, `onnx/model_q4.onnx`, and its
  ONNX weight sidecar. The real ONNX Transformers.js
  token-classification pipeline returned an entity_group array. Synthetic test:
  `My name is Alice Smith and my email address is alice@example.com.`
  Detected `private_person` score 0.99999463558 and `private_email` score
  0.99992833535. These are actual model scores, not mocked labels. The test
  used `device:'webgpu', dtype:'q4'`.
- Example smoke and Playwright harness are in `examples/privacy-filter-smoke`
  and `scripts/privacy-filter-browser-smoke.mjs`.

## Code and integrity decisions

- Separate `createPrivacyFilterSubstrate` detection from pure
  `redactPrivacySpans`. Nothing happens on import; explicit
  `BrowserSubstrateRunner.load()` fetches weights.
- Constrain spans to the original string. Transformers.js 4.3.1 grouped NER
  results omit character offsets; when no offsets are given, only accept exact
  and uniquely located substrings. Reject ambiguous/repeated, missing,
  malformed, overlapping, and stale spans. Preserve outer whitespace when
  masking. Original user text is never modified.
- Local-only model execution does **not** make asset downloading offline.
- Known functional limit: Transformers.js `simple` aggregation is **not**
  the model's upstream constrained-Viterbi calibrated decoder; do not promote
  these results to a full privacy guarantee. Some identifiers will be missed.
- Tested against synthetic text only; production redaction needs review.

## Repository isolation

Worktree `/home/myrqyry/loam/.worktrees/lil-local-privacy-filter`, branch
`aster/privacy-filter-smallest` based on `bc3e04d`; original checkout had
uncommitted changes including unrelated moderation and phonon work, all left
untouched. No push or merge.

## Final post-fix real-browser run

After the whitespace-preservation correction, a second actual browser smoke
exited **0**. It selected WebGPU/Q4 (`shader-f16=false`) and returned an
`Alice Smith` span at offsets 11–22 (score 0.99992311) and an
`alice@example.com` span at 47–64 (score 0.99979717). The independently
constructed masked copy was:
`My name is [PRIVATE_PERSON] and my email address is [PRIVATE_EMAIL].`
These are true model predictions on synthetic data, not expected mocked scores.

## Final verification

- `pnpm --filter @lil-local/privacy-filter test`: **7/7 passed**.
- `pnpm --filter @lil-local/privacy-filter typecheck`: **passed**.
- `pnpm test:privacy-filter-external`: packed independent consumer install,
  TypeScript, Vite build **passed**.
- `pnpm verify`: **failed**, specifically the broad packed-consumer
  typecheck on existing `image-embedding/src/pipeline.ts:160` and
  `qwen3-tts/src/{codec,mtp,talker}.ts` TS2769 typed-array incompatibilities.
  Those sources are unchanged in this isolated branch, and the specific
  Privacy Filter packed consumer passed. Do not report the whole gate green.
