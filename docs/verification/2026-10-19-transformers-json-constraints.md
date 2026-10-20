# Browser JSON-schema constrained generation, October 19, 2026

## Motivation
The streamer's small Qwen3 0.6B model sometimes generates unusable or incomplete structured output. Transformers.js v4.3 introduced tokenizer-aware StructuredOutputProcessor. Expose that option through the shared Lil Local pipeline.

## Scope
- Upgrade @huggingface/transformers to ^4.3.1 and add @huggingface/transformers-structured-output@4.3.1.
- Extend TextGenerationConfig with optional jsonSchema. Only when configured, instantiate a fresh StructuredOutputProcessor with the model's real tokenizer and apply it via logits_processor.
- If a tokenizer is missing, fail instead of silently generating unconstrained output. Existing unconstrained generation continues unchanged.

## Evidence
- New regression tests failed before implementation, passed after (85/85 text-gen tests).
- Entire Lil Local monorepo pnpm verify passed: typechecks, tests, compatibility gates and builds.
- This validates wrapper behavior; real Qwen3 quality/throughput must be tested separately.
- JSON constraints only govern structure; source provenance, hyperlinks and timestamps still require independent checks.
