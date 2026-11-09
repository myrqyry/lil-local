<!-- meristem-template:v1 -->
# Learnings

Project-local reusable lessons. A learning should prevent future work or error, not merely narrate what happened.

Use entries like:

## YYYY-MM-DD — Learning

**Signal / failure:**  
**Mechanism:**  
**Scope:**  
**Evidence:**  
**Recommended behavior:**  
**Validation status:** observed | tested | unvalidated


## 2026-11-08 — Do not classify Gemma browser multimodal by optimized artifact

**Signal / failure:** Repeatedly declared Gemma 4 image/audio unsupported in the
browser despite an earlier successful PodQast -> Lil Local -> LiteRT-LM real
Chromium image+audio acceptance.

**Mechanism:** Outdated text-only JavaScript documentation and tests against
the smaller 2,008,432,640-byte optimized E2B -web/-gpu artifacts overrode
stronger existing evidence for the complete 2,588,147,712-byte E2B model.
The optimized files omit the vision/audio encoder submodels. The full file
contains them. Its vision runs on GPU; the audio encoder requires CPU even
when the rest of inference uses WebGPU. GPU audio fails the backend constraint.

**Scope:** Gemma 4 E2B browser inference, the shared Lil Local text-gen worker,
PodQast consumer, model manifests and all future multimodal qualification.

**Evidence:** PodQast docs/verification/2026-11-07-gemma4-full-web-vision.md
and 2026-11-07-gemma4-native-audio.md: real Yellow PNG and banana WAV.
Lil Local docs/verification/2026-11-08-gemma4-browser-integration.md
now independently reproduces full-model image/audio through the
actual playground UI with SHA-pinned weights and real Chromium.

**Recommended behavior:** Inspect the exact model filename, byte length,
encoder subgraphs, engine settings and prior production runtime receipts
before concluding a modality is impossible. Prefer demonstrated runtime
evidence over older docs, declarations, and model-card generalizations.
Do not silently replace the full model with -web or -gpu variants.
Do not ask the user to re-prove an already established capability.

**Validation status:** tested.
