# @lil-local/phonon

Browser-first, reusable **ONNX** text-to-speech adapter for
[`onnx-community/phonon-31m-en-ONNX`](https://huggingface.co/onnx-community/phonon-31m-en-ONNX).
Runs the real `flow_lm` and `mimi_decoder` stateful graphs to synthesize
24 kHz mono PCM, rather than generating placeholder audio. Includes
`BrowserSubstrateRunner` integration; **no inference server**.

## External consumer

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createPhononSubstrate } from '@lil-local/phonon'

const runner = new BrowserSubstrateRunner(createPhononSubstrate({
  backend: 'webgpu',
  assets: yourAssetResolver, // must serve voices/default.bin
}))
await runner.load()
const audio = await runner.run({ text: 'Hello from Lil Local.' })
// AudioInferenceResult: audio.samples Float32Array, audio.sampleRate 24000
await runner.dispose()
```

Advanced consumers can instantiate `new PhononPipeline({ variant: 'q4' })`
and use the `onChunk` callback to consume decoded 80 ms frames. The default
graph selection uses FP32 on WASM; on WebGPU it selects Q4F16 only when the
actual GPU adapter advertises `shader-f16`, otherwise Q4. Explicitly requesting
Q4F16 on an incompatible backend throws.

Graph filenames and weight sidecars are pinned to HF repository revision
`4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4`; they are not bundled.
The tokenizer is loaded from that revision via Transformers.js, and both
graphs default to Hugging Face HTTPS URLs; voice data is loaded through the
caller's supplied asset resolver. **The current default is not fully offline**:
to self-host every component, a subsequent tokenizer/graph asset-routing
implementation and browser qualification are necessary. Model downloads are
initiated only when `load()` is invoked, never on import or construction.

## Evidence and limitations

- `pnpm --filter @lil-local/phonon test`
- `pnpm --filter @lil-local/phonon typecheck`
- `pnpm test:phonon-external` (packed external Vite consumer)
- `pnpm test:compatibility` (broader shared workspace gate)
- `pnpm verify` (authoritative full gate)

The **real** Chromium smoke in `examples/phonon-smoke` and
`scripts/phonon-browser-smoke.mjs` downloaded both ONNX graphs, loaded them
with ONNX Runtime Web, generated 105,600 nonzero PCM samples (4.4s at 24 kHz)
and wrote `/tmp/lil-local-phonon-browser.wav` on the test machine.
The WebGPU execution provider was requested; this does **not** prove all nodes
ran on GPU. The sample has not been independently human/ASR verified for
intelligibility. Throughput, warm/cold latency, peak GPU memory, and long-form
stability remain unqualified.

The first Q4F16 real run failed in Chromium without `shader-f16`; Q4 passed.
The pre-existing workspace packed-consumer gate currently has unrelated
TypeScript typed-array errors in `image-embedding` and `qwen3-tts`.
Do not call the entire workspace gate green unless its own run passes.
