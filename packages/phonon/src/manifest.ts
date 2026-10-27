import { type ModelAsset, type ModelManifest } from '@lil-local/inference-core';

export const PHONON_MODEL_ID = 'phonon-31m-en-ONNX';
export const PHONON_REVISION = 'main';

/** Base URL for resolving phonon model assets from the Hugging Face Hub. */
export const PHONON_HF_BASE =
  'https://huggingface.co/onnx-community/phonon-31m-en-ONNX/resolve/main/';

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
  backends: { wasm: true },
  memory: {
    downloadBytes: 41_000_000,
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
