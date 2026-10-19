# playground

React + Vite + Tailwind model lab. Consumes `@lil-local/runtime-litert` for all LiteRT loading — never import `@litertjs/core` directly.

## Key constraints

- **Build output**: `apps/playground/dist`. Vercel config (`vercel.json`) sets `buildCommand: "pnpm --filter playground build"` and `outputDirectory: "apps/playground/dist"`.
- **Runtime contract**: `useModelRunner.ts` must import `createLiteRtRuntime` from `@lil-local/runtime-litert`. Boundary tests enforce this.
- **Other browser engines**: Keep `useModelRunner.ts` on the proven LiteRT path; additional engines should use the substrate-neutral runner with a real model-specific adapter under `src/substrates`. The D1 ONNX panel is the first such consumer.
- **Dev server**: `pnpm dev` starts Vite from the workspace root.

## Verification

```bash
pnpm --filter playground typecheck
pnpm --filter playground test
pnpm --filter playground test:watch   # watch mode
```
