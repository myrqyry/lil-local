# Package revision policy

Consuming applications must use one **Lil Local** Git revision for all shared
inference packages. This keeps package contracts, runtime behavior, model
manifests, and qualification metadata aligned.

The repository was previously named `myrqyry/litert-playground`, then
`myrqyry/lil-tart`, and is now `myrqyry/lil-local`. This change also
renames the shared package namespace from `@litert-playground/*` to
`@lil-local/*`. Older pinned commits retain their older package names and
may continue to be consumed by existing applications without an automatic
migration. Historical `@litert-playground/*` packages belong to their old pinned SHA;
GitHub repository URL redirects do not change those package manifests. Do **not** mix old and new namespace packages in one consumer.

## Rules

- Select one Lil Local Git SHA for each consuming application.
- Resolve every Lil Local package used by that application from the same SHA.
- Use the canonical repository name `myrqyry/lil-local` in new dependency specs.
- Do not mix packages from different Lil Local revisions.
- Use SHA pinning as the supported revision mechanism today.
- Treat peer dependency ranges as compatibility information, not as permission
  to mix revisions.
- Do not import files from a package's `src` directory; consume the public
  `.` entrypoint.
- Future release tags may improve ergonomics, but they do not change the
  single-revision rule.

A Git-pinned consumer entry looks like:

```json
{
  "@lil-local/inference-core": "github:myrqyry/lil-local#<sha>&path:packages/inference-core",
  "@lil-local/runtime-litert": "github:myrqyry/lil-local#<sha>&path:packages/runtime-litert",
  "@lil-local/browser-cache": "github:myrqyry/lil-local#<sha>&path:packages/browser-cache"
}
```

Every entry in one application should use the same `<sha>`.

## Supported packed compatibility surface

`pnpm test:compatibility` packs and consumes the packages below as an external
Vite/TypeScript application. A package is not part of the supported downstream
surface merely because it exists in the workspace; it belongs here only when the
compatibility harness can install, type-check, and build it from its public
entrypoint.

- `@lil-local/inference-core`
- `@lil-local/runtime-litert`
- `@lil-local/browser-cache`
- `@lil-local/text-gen`
- `@lil-local/embedding-litertlm` (opt-in WebGPU text embeddings; runtime/model qualification pending)
- `@lil-local/encoder`
- `@lil-local/retrieval`
- `@lil-local/kokoro`
- `@lil-local/qwen3-tts`
- `@lil-local/image-embedding`
- `@lil-local/video-classification`
- `@lil-local/depth-estimation`
- `@lil-local/image-segmentation`

Subpath entrypoints are part of the supported surface on the same terms as the
`.` entrypoint: a subpath is supported only when the compatibility harness
resolves it through the packed tarball. Consumers must not reach past a published
subpath into a package's `src` directory.

- `@lil-local/retrieval/scoring` — late-interaction scoring without the
  ColBERT pipeline entrypoint

For the local conversational-inference stack specifically, the reusable boundary
is:

```text
inference-core
├─ runtime-litert
├─ browser-cache
├─ text-gen
├─ encoder
├─ retrieval
├─ kokoro
└─ qwen3-tts
```

Shared packages own reusable worker transport and runtime mechanics when those
mechanics are part of the supported inference surface. Product applications
still own when and why inference runs, model/persona selection, UI state,
conversation and product semantics, audio playback policy, and app-specific
persistence. Those product concerns should not leak back into Lil Local's
shared packages.
