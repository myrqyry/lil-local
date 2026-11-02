# Small local model title-generation smoke comparison — 2026-11-01

## Environment and scope

Ran on `yuzu-mint` via Ollama against 5 fixed conversation-summary prompts, each model sequentially, using a system instruction to return an accurate 3–6 word chat title, temperature 0.1 and `num_predict: 32`. These are exploratory observations, **not** a reliable quality or speed benchmark. No Lil Local browser inference was run and no product integration was made.

## Models

- `hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0` (~144 MB)
- `hf.co/mradermacher/tinybrainbot-360m-a75m-v5.1-instruct-GGUF:Q8_0` (408 MB)
- `hf.co/LiquidAI/LFM2.5-VL-450M-GGUF:Q4_K_M` (332 MB)

## Observed behavior

| Summary topic | SmolLM2 | TinyBrainBot | LFM2.5-VL |
|---|---|---|---|
| Tiny models / Lil Local | Repeated entire prompt | Repeated prompt | `Lil Local Language Models for Ideation, Chat Titles, and Routing` |
| Android background audio | Repeated entire prompt | Repeated prompt | `Android App Audio Playback Issues` |
| Retro camera shaders/timeline | Intro plus longer restatement | `Revive Retro Canyon Trails: A Comprehensive Guide for Advanced Visualization and Analysis` | `Retro Camera Shader Trails & Timeline Plugin UI` |
| Music organization on Linux | Began irrelevant apt-get instructions | `Unleash Your Creativity: A Comprehensive Music Collection for Linux!` | `Music Collection on Linux` |
| LiteRT builds but no inference | Repeated entire prompt | Began troubleshooting instructions | `LiteRT model fails inference but runs inference.` |

Warm wall-clock response times excluding first model load:
- SmolLM2: 38, 79, 88, 48 ms
- TinyBrainBot: 33, 48, 41, 65 ms
- LFM2.5-VL: 28, 44, 22, 37 ms

First calls: 1346 ms SmolLM2, 607 ms TinyBrainBot, 9085 ms LFM2.5-VL. LFM2.5 cold start included ~4026 ms load and ~5018 ms prompt evaluation. The reason for the exceptionally large first prompt evaluation cost was not diagnosed. Its instruction compliance was meaningfully better on these examples, but one output exceeded the word limit and the LiteRT title contradicted the source. Quality was judged informally rather than from a preregistered rubric.

## Decision

Do not adopt SmolLM2 or TinyBrainBot as chat-title generators from this evidence. Existing LFM2.5-VL looks promising for the use case **when already warm**, but do not introduce a new cold model load solely for titles without an end-to-end product latency/memory study. Compare against the primary assistant's own title generation and a deterministic first-message fallback. No runtime or app edits made.
