import { type ModelAsset, type ModelManifest } from '@lil-local/inference-core';

export const PHONON_MODEL_ID = 'phonon-31m-en-ONNX';
export const PHONON_REVISION = '4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4';

/** Base URL for resolving phonon model assets from the Hugging Face Hub. */
export const PHONON_HF_BASE =
  'https://huggingface.co/onnx-community/phonon-31m-en-ONNX/resolve/' + PHONON_REVISION + '/';

/** ONNX graph variants. q4f16 is the default: smallest download, fp16 activations. */
export type PhononVariant = 'fp32' | 'q4' | 'q4f16';

export interface PhononGraphSet {
  flowLm: string;
  flowLmData?: string;
  mimiDecoder: string;
  mimiDecoderData?: string;
}

export const PHONON_VARIANT_GRAPHS: Record<PhononVariant, PhononGraphSet> = {
  fp32: {
    flowLm: 'onnx/flow_lm.onnx',
    mimiDecoder: 'onnx/mimi_decoder.onnx',
  },
  q4: {
    flowLm: 'onnx/flow_lm_q4.onnx',
    flowLmData: 'onnx/flow_lm_q4.onnx_data',
    mimiDecoder: 'onnx/mimi_decoder_q4.onnx',
  },
  q4f16: {
    flowLm: 'onnx/flow_lm_q4f16.onnx',
    flowLmData: 'onnx/flow_lm_q4f16.onnx_data',
    mimiDecoder: 'onnx/mimi_decoder_q4f16.onnx',
    mimiDecoderData: 'onnx/mimi_decoder_q4f16.onnx_data',
  },
};

export const PHONON_DEFAULT_VARIANT: PhononVariant = 'q4f16';

export const phononManifest: ModelManifest = {
  modelId: PHONON_MODEL_ID,
  name: 'Phonon 31M English TTS',
  version: '1.0.0',
  capabilities: ['text-to-speech'],
  backends: { wasm: true, webgpu: true },
  memory: {
    downloadBytes: 41_000_000, // q4f16 only; fp32 needs ~161 MB
    residentBytes: 70_000_000,
  },
  assets: [
    { id: 'flow-lm', path: PHONON_VARIANT_GRAPHS[PHONON_DEFAULT_VARIANT].flowLm },
    { id: 'flow-lm-data', path: PHONON_VARIANT_GRAPHS[PHONON_DEFAULT_VARIANT].flowLmData ?? '' },
    { id: 'mimi-decoder', path: PHONON_VARIANT_GRAPHS[PHONON_DEFAULT_VARIANT].mimiDecoder },
    { id: 'mimi-decoder-data', path: PHONON_VARIANT_GRAPHS[PHONON_DEFAULT_VARIANT].mimiDecoderData ?? '' },
    { id: 'tokenizer', path: 'tokenizer.json' },
    { id: 'default-voice', path: 'voices/default.bin' },
  ],
};

/** A runtime receipt must describe the graph variant actually loaded. */
export function manifestForPhononVariant(variant: PhononVariant): ModelManifest {
  const graphs = PHONON_VARIANT_GRAPHS[variant];
  const assets: ModelAsset[] = [
    { id: 'flow-lm', path: graphs.flowLm },
    ...(graphs.flowLmData ? [{ id: 'flow-lm-data', path: graphs.flowLmData }] : []),
    { id: 'mimi-decoder', path: graphs.mimiDecoder },
    ...(graphs.mimiDecoderData ? [{ id: 'mimi-decoder-data', path: graphs.mimiDecoderData }] : []),
    { id: 'tokenizer', path: 'tokenizer.json' },
    { id: 'default-voice', path: 'voices/default.bin' },
  ];
  const bytes = variant === 'fp32' ? 161_000_000 : variant === 'q4' ? 67_000_000 : 41_000_000;
  return { ...phononManifest, assets,
    memory: { downloadBytes: bytes, residentBytes: Math.max(bytes, 70_000_000) } };
}
