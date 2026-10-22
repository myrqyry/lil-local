import { describe, expect, it } from 'vitest'
import {
  decodeSamMaskCandidates,
  getSam2ModelVariant,
  normalizeFaceMaskLabel,
  validateSegmentationSource,
  validatePointInImage,
  FACE_PARSING_MODEL,
  SAM2_MODEL_VARIANTS,
} from './index'

describe('portable image-segmentation contracts', () => {
  it('pins only validated on-demand browser model variants', () => {
    expect(FACE_PARSING_MODEL.id).toBe('Xenova/face-parsing')
    expect(FACE_PARSING_MODEL.device).toBe('wasm')
    expect(FACE_PARSING_MODEL.dtype).toBe('q8')
    expect(Object.keys(SAM2_MODEL_VARIANTS)).toEqual(['tiny', 'small'])
    expect(getSam2ModelVariant('tiny').repo).toBe('onnx-community/sam2.1-hiera-tiny-ONNX')
    expect(getSam2ModelVariant('small').revision).toBe('a7df49d8de14b9d2e4504d1687b0d568f905fd8d')
    expect(() => getSam2ModelVariant('../untrusted')).toThrow(/unsupported/i)
  })
  it('exposes known anatomical labels only', () => {
    expect(normalizeFaceMaskLabel('l_eye.png')).toBe('l_eye')
    expect(normalizeFaceMaskLabel('background')).toBeNull()
    expect(normalizeFaceMaskLabel('bad.svg')).toBeNull()
  })
  it('ranks mask candidates and rejects invalid payloads', () => {
    const masks = decodeSamMaskCandidates(
      Uint8Array.from([0,1,0,0, 1,0,1,1, 1,1,0,1]),
      [1, 3, 2, 2], Float32Array.from([0.21,0.91,0.49]),
    )
    expect(masks).toHaveLength(3)
    expect(masks[0]).toMatchObject({ label:'sam-object-1', width:2, height:2 })
    expect(masks[0].score).toBeCloseTo(.91)
    expect(Array.from(masks[0].data)).toEqual([255,0,255,255])
    expect(() => decodeSamMaskCandidates(new Uint8Array(12), [1,3,2,2], Float32Array.from([0,0,0]))).toThrow(/confiden/i)
    expect(() => decodeSamMaskCandidates(new Uint8Array(11), [1,3,2,2], Float32Array.from([1,1,1]))).toThrow(/length/i)
    expect(() => decodeSamMaskCandidates(new Uint8Array(12), [3,1,2,2], Float32Array.from([1,1,1]))).toThrow(/shape/i)
    expect(() => decodeSamMaskCandidates(new Uint8Array(12), [1,3,2,2], Float32Array.from([NaN,1,1]))).toThrow(/score/i)
  })
  it('rejects oversized and untrusted inputs before model loading', () => {
    expect(() => validateSegmentationSource(new Blob(['no'],{type:'text/plain'}))).toThrow(/supported/i)
    expect(() => validateSegmentationSource(new Blob([new Uint8Array(21 * 1024 * 1024)], {type:'image/png'}))).toThrow(/20 MiB/)
    expect(() => validatePointInImage({x:-1,y:0},384,480)).toThrow(/outside/i)
    expect(() => validatePointInImage({x:0,y:0},384,480)).not.toThrow()
    expect(() => validatePointInImage({x:3,y:3},0,4)).toThrow(/image/i)
  })
})
