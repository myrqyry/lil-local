# video-classification

MoViNet video classification pipeline.

## Key constraints

- **Transactional frame commit**: `buildInputTensors` must not mutate `state.frameNum`. It returns `{ inputs, nextFrame }` where `nextFrame = frameNum + 1`. `run()` calls `state.commitFrame(nextFrame)` only after `updateState` succeeds. There is no rollback path; `discardFrame()` was removed.
- **SIMD probe correctness**: the `v128.const` WASM instruction requires a 16-byte immediate. `probeWasmSimd()` in `runtime-litert` must carry the full 16 bytes and the code section size must match (`0x16`). An undersized immediate causes `WebAssembly.validate` to return `false` on SIMD-capable browsers.

## Verification

```bash
pnpm --filter @lil-local/video-classification typecheck
pnpm --filter @lil-local/video-classification test
```
