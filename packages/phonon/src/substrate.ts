import type { AssetResolver, AudioInferenceResult, BrowserSubstrate, PipelineProgress } from '@lil-local/inference-core';
import { PhononPipeline, type PhononInput, type PhononPipelineOptions } from './pipeline';

export interface PhononSubstrateOptions extends PhononPipelineOptions {
  backend: 'webgpu' | 'wasm';
  assets: AssetResolver;
}

/** Inert browser-substrate adapter; does not require a LiteRT runtime or a service. */
export function createPhononSubstrate(options: PhononSubstrateOptions):
  BrowserSubstrate<PhononInput, AudioInferenceResult, PipelineProgress> {
  return {
    id: 'phonon-31m-en-onnx',
    format: 'onnx',
    capabilities: ['text-to-speech'],
    requestedBackends: [options.backend],
    async create(onProgress, signal) {
      const pipeline = new PhononPipeline(options);
      pipeline.onProgress = onProgress;
      await pipeline.load({ backend: options.backend, assets: options.assets, signal });
      return { infer: input => pipeline.run(input), dispose: () => pipeline.dispose() };
    },
  };
}
