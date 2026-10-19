# qwen3-tts

Qwen3-TTS pipeline: three host-orchestrated LiteRT graphs (talker, MTP, codec) via Web Workers.

## Key constraints

- **Browser memory is a known limitation**: the model set exceeds the practical browser WASM/JS budget during prefill. The `browserMemory` manifest variant (`mtp_folded_int8`) and short-KV talker exports are compatibility probes for future LiteRT.js improvements, not a working browser path.
- **Worker lifecycle**: generator and decoder workers must be terminated in `finally` blocks so a failed generation/decoding does not strand workers.
- **Peer deps**: `@lil-local/inference-core` and `@lil-local/runtime-litert` at `0.1.x`.

## Verification

```bash
pnpm --filter @lil-local/qwen3-tts typecheck
pnpm --filter @lil-local/qwen3-tts test
```
