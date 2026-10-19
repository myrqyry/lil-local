# D1 Omni 600M ONNX WebGPU — opt-in text decision slice

Status: **real text-choice inference verified once on NVIDIA/Turing WebGPU** (2026-10-12); not calibrated, production-qualified, or verified for vision/audio.
This is a separate runtime family from Lil Local's LiteRT.js inference, and
separate from Veyax's existing native GGUF/llama.cpp visual proof.

## Source and scope

- Model: [onnx-community/d1-omni-600M-ONNX](https://huggingface.co/onnx-community/d1-omni-600M-ONNX)
- Pinned revision: `4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0`
- Upstream question framing: [LiquidAI/d1-omni-600M/prompt.py](https://huggingface.co/LiquidAI/d1-omni-600M/blob/main/prompt.py)
- Graphs: `onnx/model_quantized.onnx` + `onnx/model_quantized.onnx_data` (approx. 406 MB of external weights; additional tokenizer assets)
- Execution: ONNX Runtime Web, **WebGPU requested explicitly**, no implicit fallback configured
- Questions: text state + one `choice` (2–16 options) or `noul` yes/no statement per run; never generation.
- Score processing: source-trained question-count temperatures; softmax across all options, with `noul` option order `false`, `true`. The UI reports raw **uncalibrated** probabilities, not decision authorization.
- Download: only after Load click. Text input stays in the browser. Model assets are read from Hugging Face; weight caching is managed by the browser/ONNX stack, not Lil Local's LiteRT model library.
- **No image/audio:** the upstream ONNX model includes vision and audio encoder graphs; this deliberately scoped slice does not implement them.
- License: weights are © Liquid AI, Inc. under **LFM Open License v1.0**; check its attribution and commercial-use terms before redistribution. Our adapter code is Apache-2.0.

### Manual browser check

1. Run `pnpm dev`; choose **D1 Omni 600M (WebGPU Decisions)** in the model list.
2. Check **no model download occurs** until clicking **Load WebGPU model**.
3. Load in a WebGPU-capable Chromium browser with sufficient VRAM. No fallback is attempted when WebGPU is unavailable.
4. Ask a choice and a yes/no question. Observe actual logits/probabilities and request latency.
5. Compare top choices and probabilities against Veyax's previously verified GGUF Q8 server on matching prompts. Treat formats and question framings as potentially different until measured.
6. Unload and verify the browser session is released.

Neither passing a typecheck nor successful graph compilation counts as successful ONNX inference. The backend field records the **requested** WebGPU EP; ONNX Runtime may route individual operators differently, which must be checked separately for full-GPU claims.

### Repeatable checks

```bash
pnpm --filter @lil-local/decision-onnx typecheck
pnpm --filter @lil-local/decision-onnx test
pnpm --filter playground typecheck
pnpm verify
```

The actual Chrome/Q8/WebGPU-requested model loaded and produced a choice result on NVIDIA Turing on 2026-10-12 (see [observed receipt](verification/2026-10-12-d1-webgpu-text.md)); ORT warned some nodes may run on CPU. Numerical parity vs GGUF, lifecycle failures during downloads, and reliability/latency distribution remain **unverified**. In particular, the upstream `open-jev` 0.1.2 library does not list this D1 family in its public registry, so this code drives the documented ONNX inputs directly rather than inventing support in that package.
