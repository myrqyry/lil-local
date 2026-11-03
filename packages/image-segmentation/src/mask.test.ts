import { describe, expect, it } from 'vitest'
import { prepareSaliencyInput, decodeSaliencyMask, extractMaskLayer } from './mask'

describe('foreground mask portable contract', () => {
  it('prepares NCHW RGB with image-max scaling (not a fake monochrome input)', () => {
    const image = { width: 2, height: 1,
      data: new Uint8ClampedArray([255, 127, 0, 255, 0, 64, 128, 255]) }
    const prepared = prepareSaliencyInput(image, 2, 1)
    expect(prepared).toHaveLength(6)
    expect(prepared[0]).toBeCloseTo((1 - .485) / .229)
    expect(prepared[1]).toBeCloseTo((0 - .485) / .229)
    expect(prepared[2]).toBeCloseTo((127 / 255 - .456) / .224)
    expect(prepared[3]).toBeCloseTo((64 / 255 - .456) / .224)
    expect(prepared[4]).toBeCloseTo((0 - .406) / .225)
    expect(prepared[5]).toBeCloseTo((128 / 255 - .406) / .225)
  })

  it('uses bilinear resampling and enforces bounded source dimensions', () => {
    const image = { width: 2, height: 1,
      data: new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 255]) }
    const resized = prepareSaliencyInput(image, 3, 1)
    expect(resized[1]).toBeCloseTo((.5 - .485) / .229)
    expect(resized[7]).toBeCloseTo((.5 - .406) / .225)
    expect(() => prepareSaliencyInput({width:2,height:2,data:new Uint8ClampedArray(4)},2,2)).toThrow(/RGBA/)
    expect(() => prepareSaliencyInput({width:8192,height:8192,data:new Uint8ClampedArray(0)},2,2)).toThrow(/pixels/)
  })

  it('decodes soft alpha without pretending it is a part-label map', () => {
    const frame = decodeSaliencyMask(new Float32Array([0, .5, 1, .25]), 2, 2, 8, 4)
    expect(frame).toMatchObject({ kind: 'foreground-mask', width:2, height:2, sourceWidth:8, sourceHeight:4 })
    expect(Array.from(frame.alpha)).toEqual([0,128,255,64])
    expect(() => decodeSaliencyMask(new Float32Array([0, NaN]),2,1,1,1)).toThrow(/finite/)
    expect(() => decodeSaliencyMask(new Float32Array([0]),2,1,1,1)).toThrow(/length/)
  })

  it('extracts alpha-premultiplied transparent RGBA crop in source coordinates', () => {
    const image = {width:3,height:2,data:new Uint8ClampedArray([
      100,110,120,255, 130,140,150,255, 160,170,180,255,
      200,210,220,255, 230,240,250,128, 250,240,230,255,
    ])}
    const mask = decodeSaliencyMask(new Float32Array([0,0,0,0,0.5,1]),3,2,3,2)
    const result = extractMaskLayer(image,mask)
    expect(result).toMatchObject({x:1,y:1,width:2,height:1})
    expect(Array.from(result.data)).toEqual([230,240,250,64,250,240,230,255])
    expect(image.data[19]).toBe(128) // source unchanged
  })

  it('rejects mismatched source/mask bounds before allocation', () => {
    const mask = decodeSaliencyMask(new Float32Array([1,1,1,1]),2,2,4,4)
    expect(() => extractMaskLayer({width:2,height:2,data:new Uint8ClampedArray(16)},mask)).toThrow(/source/)
  })
})
