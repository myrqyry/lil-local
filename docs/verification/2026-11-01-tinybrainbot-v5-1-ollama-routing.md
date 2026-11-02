# TinyBrainBot v5.1 Q8 — local routing smoke test (2026-11-01)

## Scope

Exploratory, read-only Ollama inference on yuzu-mint; **not** a browser/Lil Local inference qualification, and not an application integration. No authority over tools or execution granted.

- Model: `hf.co/mradermacher/tinybrainbot-360m-a75m-v5.1-instruct-GGUF:Q8_0`
- Ollama: 0.40.11; installed model listed at 408 MB
- Prompt: classify one user utterance as `rewrite`, `retrieve`, `tool`, `chat`, or `ambiguous`, temperature 0
- Five manually assigned expected labels; these are illustrative smoke probes, not a statistically meaningful evaluation set.

## Results

| Utterance | Expected | Unconstrained response | JSON-schema constrained label |
|---|---|---|---|
| make this paragraph shorter | rewrite | Generated unrelated report text | chat |
| what did we decide yesterday about the audio app? | retrieve | Fabricated a decision about a sound feature | ambiguous |
| Open the project settings please | tool | Explained how to open settings | tool |
| hi there! | chat | Conversational greeting | tool |
| can you make it work like the other one? | ambiguous | Began an unrelated rewrite | ambiguous |

Unconstrained: **0/5 valid labels**. With Ollama `format` JSON Schema constraining `route` to the five enum choices: **5/5 valid JSON enum outputs, 2/5 correct labels**. Schema validity is not semantic accuracy.

## Assessment

The local GGUF can load and produce text, but these results do **not** justify using it for autonomous routing. Structured output makes the shape valid but does not solve wrong classifications. The existing `decision-onnx` lane is independent and was not modified. No browser runtime, model-specific loader, or real app path was exercised. If revisited, establish a larger representative gold dataset, deterministic baseline, stronger-model comparison, and explicit abstention/escalation policy before promoting it from shadow experiments.
