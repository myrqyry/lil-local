# @lil-local/privacy-filter

Opt-in, **browser-local** personally identifiable information detector built on
[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter).
This is a **reusable Lil Local ONNX substrate**, not a server or an automatic
censorship layer.

## Consume from another application

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createPrivacyFilterSubstrate, redactPrivacySpans } from '@lil-local/privacy-filter'

const runner = new BrowserSubstrateRunner(createPrivacyFilterSubstrate({
  device: 'webgpu', // or 'wasm'
  // dtype: 'q4' | 'q4f16' | 'fp16' | 'fp32' | 'q8'
}))
await runner.load() // explicit: first call downloads the ONNX weights + tokenizer
const original = 'Please contact alice@example.com.'
const result = await runner.run({ text: original })
const copyForReview = redactPrivacySpans(original, result.spans)
// Never replaces original automatically; share nothing without explicit approval.
await runner.dispose()
```

Default dtype selection: Q4F16 (~809 MB ONNX weights) when WebGPU advertises
`shader-f16`; otherwise Q4 (~917 MB). This is a device capability selection,
**not** a guarantee of execution on every driver. Explicit dtype options
remain available for other users. Model revision pinned at
`7ffa9a043d54d1be65afb281eddf0ffbe629385b`.
Tokenizer download is additional (~27.9 MB); runtime memory may exceed file size.
Weights and tokenizer are not bundled in the published package.

**Detection is not masking.** The caller can review categories, confidence,
exact source offsets, and separately construct a redacted copy. The original
string is never modified. Detection covers eight fixed categories:
`account_number`, `private_address`, `private_date`, `private_email`,
`private_person`, `private_phone`, `private_url`, `secret`.

### Crucial limitations

- The Transformers.js 4.3.1 `aggregation_strategy: 'simple'` path is **not**
  equivalent to the model's reference constrained-Viterbi decoder and tuned
  operating points. The reference decoder remains a separate qualification task.
- Transformers.js currently omits character offsets from grouped token results.
  `resolvePrivacySpans` only aligns an **exact, unique** substring of the
  original; any normalization mismatch, duplicate, invalid output or overlap
  throws instead of quietly masking the wrong text. Whitespace outside spans
  remains intact. Legitimate repeated names can therefore need manual review.
- It can miss secrets and misclassify content. This is **not** anonymization,
  regulatory compliance, or a guarantee that sharing the result is safe.
- Explicit model loading contacts Hugging Face to download assets, but model
  inference itself runs in the chosen local browser execution provider. This is
  not yet a fully self-hosted/offline model-assets setup.
- Model routing through other user apps, live large-text performance, reference
  Viterbi parity, and formal redaction effectiveness remain unqualified.

### Verification

`pnpm --filter @lil-local/privacy-filter test`
`pnpm --filter @lil-local/privacy-filter typecheck`
`pnpm test:privacy-filter-external`

Real Chromium smoke script:
`pnpm exec vite --config examples/privacy-filter-smoke/vite.config.ts`
then `node scripts/privacy-filter-browser-smoke.mjs`.
This runs an intentionally synthetic name/email in a browser; it does not
inspect or upload private user data. See
`docs/verification/2026-10-26-openai-privacy-filter-q4.md`.
