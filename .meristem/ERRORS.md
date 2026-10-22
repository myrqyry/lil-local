<!-- meristem-template:v1 -->
# Errors

Preserve costly, recurring, or non-obvious failures that future agents could plausibly repeat.

Use entries like:

## YYYY-MM-DD — Error / regression

**Observed failure:**  
**Root mechanism:**  
**Affected environment:**  
**Recovery / fix:**  
**How to detect earlier:**  
**Evidence / references:**

## 2026-10-21 — Bonsai 2B CPU-native PrismML load assertion

**Observed:** PrismML official release `prism-b10787-73bfe6c`,
the exact SHA-256 verified 2B PTQ1_0 GGUF, isolated low-priority
`llama-cli` with `-ngl 0`, 256-token context, 2 CPU threads and a 75-second
timeout: exit 134, `GGML_ASSERT(ggml_is_padded_1d(a))` in
`ggml_scale_impl` at `ggml.c:3413` during model load. No token generated.
No GPU inference or machine instability observed.

**Consequence:** Native Bonsai 2B model is not runtime-qualified. Do not
list it as working because the hash matched or the native executable
reported help correctly. The separate ONNX decoder has 18 Scan nodes and
is structurally unsuitable for the current ORT WebGPU path.

**Evidence:** `docs/verification/2026-10-21-bonsai2-candidate-qualification.md`.
No speculative parameter retries after this failure.

**Resolution:** open; requires root-cause investigation against PrismML
loader/model and a real bounded generation result.
