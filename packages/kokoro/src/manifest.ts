import { type ModelAsset, type ModelManifest } from '@lil-local/inference-core';

export const kokoroManifest: ModelManifest = {
  modelId: "kokoro-82m-v1.0",
  name: "Kokoro 82M v1.0",
  version: "1.0.0",
  capabilities: ["text-to-speech"],
  backends: { wasm: true },
  memory: {
    downloadBytes: 329_000_000,
    residentBytes: 330_000_000,
  },
  assets: [
    {
      id: "model",
      path: "onnx-community/Kokoro-82M-v1.0-ONNX/onnx/model_quantized.onnx",
      bytes: 329_000_000,
    },
  ],
};
