/** Audited, immutable HF revisions. Assets never ship with this package. */
export const FACE_PARSING_MODEL = {
  id: 'Xenova/face-parsing',
  revision: 'f25b9b521a8783d4e78e80e026ef4c2a15f821e0',
  dtype: 'q8',
  device: 'wasm',
} as const

export const SAM2_MODEL_VARIANTS = {
  tiny: {
    repo: 'onnx-community/sam2.1-hiera-tiny-ONNX',
    revision: '814a066640debee5a91e70aa401fb8e17e030503',
    dtype: 'q8',
    label: 'SAM 2.1 Tiny · lighter',
  },
  small: {
    repo: 'onnx-community/sam2.1-hiera-small-ONNX',
    revision: 'a7df49d8de14b9d2e4504d1687b0d568f905fd8d',
    dtype: 'q8',
    label: 'SAM 2.1 Small · more detail',
  },
} as const
export type Sam2Variant = keyof typeof SAM2_MODEL_VARIANTS
export function getSam2ModelVariant(input: string) {
  if (!Object.prototype.hasOwnProperty.call(SAM2_MODEL_VARIANTS, input)) {
    throw new Error('Unsupported SAM model variant')
  }
  return SAM2_MODEL_VARIANTS[input as Sam2Variant]
}
