import type { ModelManifest } from '@lil-local/inference-core'

const REPO = 'litert-community/LFM2.5-ColBERT-350M'

export const colbertManifest: ModelManifest = {
  modelId: 'lfm2.5-colbert-350m',
  name: 'LFM2.5 ColBERT-350M',
  version: '2.5.0',
  capabilities: ['multi-vector-embedding', 'reranking'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 700_000_000, residentBytes: 700_000_000 },
  assets: [
    { id: 'model', path: `${REPO}/resolve/main/LFM2.5-ColBERT-350M_fp16.tflite` },
    { id: 'tokenizer', path: `${REPO}/resolve/main/tokenizer.json` },
  ],
  verification: { assets: 'untested', compile: 'untested', inference: 'untested', output: 'untested' },
}

export const colbertWi8fcManifest: ModelManifest = {
  modelId: 'lfm2.5-colbert-350m-wi8fc',
  name: 'LFM2.5 ColBERT-350M (weight-int8, full-channel)',
  version: '2.5.0',
  capabilities: ['multi-vector-embedding', 'reranking'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 350_000_000, residentBytes: 350_000_000 },
  assets: [
    { id: 'model', path: `${REPO}/resolve/main/LFM2.5-ColBERT-350M_wi8fc.tflite` },
    { id: 'tokenizer', path: `${REPO}/resolve/main/tokenizer.json` },
  ],
  verification: { assets: 'untested', compile: 'untested', inference: 'untested', output: 'untested' },
}
