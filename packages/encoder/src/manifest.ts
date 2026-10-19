import type { ModelManifest } from '@lil-local/inference-core'

const REPO_230M = 'litert-community/LFM2.5-Encoder-230M'
const REPO_SPELLCHECKER = 'litert-community/LFM2.5-Encoder-350M-Spellchecker'
const REPO_POLICY_LINTER = 'litert-community/LFM2.5-Encoder-350M-Policy-Linter'

export const encoder230mManifest: ModelManifest = {
  modelId: 'lfm2.5-encoder-230m',
  name: 'LFM2.5 Encoder-230M',
  version: '2.5.0',
  capabilities: ['text-embedding'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 230_000_000, residentBytes: 230_000_000 },
  assets: [
    { id: 'model', path: `${REPO_230M}/resolve/main/LFM2.5-Encoder-230M_fp16.tflite` },
    { id: 'tokenizer', path: `${REPO_230M}/resolve/main/tokenizer.json` },
  ],
  verification: { assets: 'untested', compile: 'untested', inference: 'untested', output: 'untested' },
}

export const encoderSpellcheckerManifest: ModelManifest = {
  modelId: 'lfm2.5-encoder-350m-spellchecker',
  name: 'LFM2.5 Encoder-350M Spellchecker',
  version: '2.5.0',
  capabilities: ['token-classification'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 350_000_000, residentBytes: 350_000_000 },
  assets: [
    { id: 'model', path: `${REPO_SPELLCHECKER}/resolve/main/LFM2.5-Encoder-350M-Spellchecker_fp16.tflite` },
    { id: 'tokenizer', path: `${REPO_SPELLCHECKER}/resolve/main/tokenizer.json` },
  ],
  verification: { assets: 'untested', compile: 'untested', inference: 'untested', output: 'untested' },
}

export const encoderPolicyLinterManifest: ModelManifest = {
  modelId: 'lfm2.5-encoder-350m-policy-linter',
  name: 'LFM2.5 Encoder-350M Policy-Linter',
  version: '2.5.0',
  capabilities: ['token-classification', 'policy-classification'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 350_000_000, residentBytes: 350_000_000 },
  assets: [
    { id: 'model', path: `${REPO_POLICY_LINTER}/resolve/main/LFM2.5-Encoder-350M-Policy-Linter_fp16.tflite` },
    { id: 'tokenizer', path: `${REPO_POLICY_LINTER}/resolve/main/tokenizer.json` },
  ],
  verification: { assets: 'untested', compile: 'untested', inference: 'untested', output: 'untested' },
}

export type EncoderCapability = 'text-embedding' | 'token-classification' | 'policy-classification'

export function selectEncoderManifest(capability: EncoderCapability): ModelManifest {
  switch (capability) {
    case 'text-embedding':
      return encoder230mManifest
    case 'policy-classification':
      return encoderPolicyLinterManifest
    case 'token-classification':
      return encoderSpellcheckerManifest
  }
}
