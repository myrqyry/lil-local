# @lil-local/nemotron-asr

Opt-in, reusable browser ONNX speech recognition for **Nemotron 3.5 ASR
Streaming 0.6B INT4**. This is the original Lil Local playground inference
implementation moved into a public package, not a second model runner.

Includes the MIT-licensed upstream-adapted model driver, log-mel frontend,
worker entrypoint, and typed client. See `LICENSE.upstream.txt`,
`docs/verification/2026-11-02-nemotron-browser-asr.md` in the repository,
and the original project at People-11/Nemotron_Web. **The NVIDIA model files
have separate licensing terms** and are not bundled with this package.

## Public API

```ts
import {
  NemotronClient, decodeAudioFile, logMelSpectrogram,
  CHUNK_SAMPLES, validateAudioSamples,
} from '@lil-local/nemotron-asr'

const client = new NemotronClient({
  assets: {
    // Required host asset unless using the default playground URL:
    jointModelUrl: '/models/nemotron/joint-argmax.onnx',
    // Optional override: modelRoot: '/cached/nemotron-model',
  },
})
try {
  // No weights/worker are downloaded during module import or catalog discovery.
  await client.load(console.info, 'wasm') // explicit user action
  const pcm = await decodeAudioFile(selectedFile)
  validateAudioSamples(pcm)
  const padded = Math.ceil(pcm.length / CHUNK_SAMPLES) * CHUNK_SAMPLES
  const mel = logMelSpectrogram(pcm, padded)
  const text = await client.transcribe(mel, 101, console.info)
  console.log(text)
} finally {
  await client.dispose()
}
```

The original playground hosts `joint-argmax.onnx` at
`/nemotron-web/joint-argmax.onnx`; that remains the default for compatibility.
**Every external consumer must provide the same small optimized joint ONNX
graph at an accessible URL** using `assets.jointModelUrl`, and must ensure its
WebGPU/WASM ORT runtime assets can load from that origin. The pinned remote
encoder, decoder, tokenizer and `.data` files load only after `client.load()`;
`assets.modelRoot` lets a consumer use a local/cached source instead.

`client.startStream(langId)` then `client.pushAudio(mono16kSamples)` supports
worker-side streaming. `pushAudio` copies caller-owned PCM before using a
transferable; a consumer can reuse its AudioWorklet buffers safely.
The client owns its worker. Microphone or tab permission, capture/worklets,
source timestamps, episode/commentary policy and playback are **app concerns**.
There is no cloud ASR fallback. An explicit WebGPU backend request fails
closed; `auto` can fall back to WASM if GPU initialization fails.

Do not assume that the bare worker has saved output, disk persistence, model
integrity checking, or native cancellation beyond its implemented API.

## Verification boundaries

- Reuses previously verified original Chromium ONNX transcript path (see
  `docs/verification/2026-11-02-nemotron-browser-asr.md`).
- `pnpm --filter @lil-local/nemotron-asr test` and `typecheck` check the public
  interface and worker message contracts without loading the model.
- `pnpm test:compatibility` packs, installs, imports and builds the public
  package in a separate consumer.
- Actual Chromium was used to instantiate and dispose a real worker from the
  relocated package with no HTTP requests outside localhost.
- **Not yet requalified:** a speech transcription using the full ONNX model
  through the relocated worker, microphone recording after migration, other
  platforms, and performance / memory in sustained sessions.
