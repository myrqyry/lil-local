# runtime-litert

Managed LiteRT runtime wrapping `@litertjs/core`.

## Key constraints

- **Backend order**: WebGPU → WebNN → WASM. Strict when explicitly requested.
- **WASM capability probes are real**: `probeWasmSimd()` uses a validated SIMD v128 module (must carry the 16-byte immediate for `v128.const`; code section size must match). `probeWasmJspi()` checks `WebAssembly.promising` + `WebAssembly.Suspending`, not `WebAssembly.Function`.
- **Coordinator `emit()`**: each listener wrapped in `try/catch`; one throwing listener must not poison the inference it observes.
- **Peer dependency**: `@lil-local/inference-core` at `0.1.x`. Do not introduce model-package deps here.

## Verification

```bash
pnpm --filter @lil-local/runtime-litert typecheck
pnpm --filter @lil-local/runtime-litert test
```

`capabilities.test.ts` should cover SIMD/threads/JSPI probing behavior, not just ordering.
