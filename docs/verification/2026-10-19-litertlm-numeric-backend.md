# LiteRT-LM browser text-generation backend mapping — 2026-10-19

A consuming app (Creator's Insight Engine) imported the actual
`@lil-local/text-gen` pipeline with a checked official Qwen3 0.6B
checkpoint (614,236,160 bytes, pinned SHA-256) and requested WebGPU
on NVIDIA Turing Chromium. The native engine failed before creating a
conversation: `wasm.ModelAssets.create is not a function`.

Root cause verified against installed `@litert-lm/core@0.18.1`:
`Engine.create()` accepts numeric SDK backends (`GPU_ARTISAN=2`,
`CPU_ARTISAN=1`). The Lil Local pipeline previously forwarded the
generic `'webgpu'`/`'wasm'` strings unchanged, so the SDK did not take
the GPU Artisan streamed-model branch and attempted the wrong
`ModelAssets.create` API.

Fix: translate generic API backends at the SDK boundary before
`Engine.create`. Preserve the public `loadForBackend('webgpu'|'wasm')`
contract and existing unset/WebNN fallback. Also remove two unused
internal timing fields and an unused Transformers context member
whose warnings surfaced in strict external TypeScript consumers;
there is no behavior tied to these unused fields.

Evidence: existing tests failed with the new numeric SDK assertions
(3/80); the finished text-gen package passed **80/80 tests** and
package TypeScript checks. Full `pnpm verify` **exit 0** (all workspace
typechecks, tests, package/boundary/compatibility/conversion/qualification
gates, builds). New consumer-native GPU inference still requires a
post-publication browser run of the real model through the fixed package;
this change alone is not a successful inference receipt.
