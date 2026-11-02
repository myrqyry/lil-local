import { describe, expect, it } from 'vitest'
import {
  PIX2PIX_EDGE2SHOES_FOLDED_SHA256,
  PIX2PIX_EDGE2SHOES_WEIGHT_BYTES,
  PIX2PIX_MAJOR_STATIC_GPU_BYTES,
  pix2pixEdge2ShoesManifest,
  nchwToRgba,
  validatePix2PixInput,
} from './contract'

describe('pix2pix WebGPU contract', () => {
  it('pins the verified folded edges2shoes artifact and WebGPU-only capability', () => {
    expect(PIX2PIX_EDGE2SHOES_WEIGHT_BYTES).toBe(217_677_836)
    expect(PIX2PIX_MAJOR_STATIC_GPU_BYTES).toBe(253_233_196)
    expect(PIX2PIX_EDGE2SHOES_FOLDED_SHA256).toBe(
      'eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34',
    )
    expect(pix2pixEdge2ShoesManifest.capabilities).toEqual(['image-edit'])
    expect(pix2pixEdge2ShoesManifest.backends).toEqual({ webgpu: 'experimental' })
    expect(pix2pixEdge2ShoesManifest.assets).toEqual([
      expect.objectContaining({
        bytes: PIX2PIX_EDGE2SHOES_WEIGHT_BYTES,
        sha256: PIX2PIX_EDGE2SHOES_FOLDED_SHA256,
      }),
    ])
    expect(pix2pixEdge2ShoesManifest.verification?.qualification).toBe('limited')
  })

  it('converts NCHW [-1,1] output to opaque RGBA8', () => {
    const plane = 4
    const values = new Float32Array([
      -1, 0, 1, 2,
      1, 0, -1, -2,
      0, 0.5, -0.5, 0,
    ])
    expect([...nchwToRgba(values, 2, 2)]).toEqual([
      0, 255, 128, 255,
      128, 128, 191, 255,
      255, 0, 64, 255,
      255, 0, 128, 255,
    ])
    expect(values.length).toBe(plane * 3)
  })

  it('accepts only an exact 256x256 RGBA byte image', () => {
    const valid = { width: 256, height: 256, data: new Uint8Array(256 * 256 * 4) }
    expect(validatePix2PixInput(valid)).toBe(valid)
    expect(() => validatePix2PixInput({ ...valid, width: 255 })).toThrow(/256x256/)
    expect(() => validatePix2PixInput({ ...valid, data: new Uint8Array(3) })).toThrow(/262,144/)
  })
})
