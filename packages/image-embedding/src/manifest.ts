import type { ModelManifest } from '@lil-local/inference-core';

const MODEL_BASE = 'https://huggingface.co/mlboydaisuke/clip-vit-b32-litert/resolve/c9253a36391d0881d4349f07d14e8b2d2054a955';

export const clipImageEmbeddingManifest: ModelManifest = {
  modelId: 'clip-vit-b32',
  name: 'CLIP ViT-B/32',
  version: '1.0.0',
  capabilities: ['image-embedding'],
  backends: { webgpu: true, wasm: true },
  memory: { downloadBytes: 180_000_000, residentBytes: 260_000_000 },
  assets: [
    {
      id: 'model',
      path: `${MODEL_BASE}/clip_image_encoder.tflite`,
      role: 'image encoder',
    },
    {
      id: 'text-embeddings',
      path: `${MODEL_BASE}/text_embeddings.bin`,
      role: 'text embedding table',
    },
    { id: 'labels', path: `${MODEL_BASE}/labels.txt`, role: 'label mapping' },
  ],
  verification: {
    assets: 'untested',
    compile: 'untested',
    inference: 'untested',
    output: 'untested',
    qualification: 'limited',
    upstreamRevision: 'c9253a36391d0881d4349f07d14e8b2d2054a955',
    environments: [
      { browser: 'CI contract tests', backend: 'webgpu', runtime: 'manifest-only' },
      { browser: 'CI contract tests', backend: 'wasm', runtime: 'manifest-only' },
    ],
    expectedOutput: {
      preprocessing: [
        'center crop input to a square',
        'resize to 224x224',
        'RGB channel order',
        'divide channel values by 255',
        'normalize with mean [0.48145466, 0.4578275, 0.40821073]',
        'normalize with standard deviation [0.26862954, 0.26130258, 0.27577711]',
      ],
      outputDimension: 512,
    },
  },
};
