import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS, createSemanticSegmenterSubstrate,
  copySemanticCategoryMask,
} from './index'

describe('MediaPipe DeepLab v3 reusable semantic segmentation', () => {
  it('pins actual official publisher bytes', () => {
    expect(MEDIAPIPE_VISION_MODELS.semantic.bytes).toBe(2_780_176)
    expect(MEDIAPIPE_VISION_MODELS.semantic.sha256)
      .toBe('ff36e24d40547fe9e645e2f4e8745d1876d6e38b332d39a82f0bf0f5d1d561b3')
    expect(MEDIAPIPE_VISION_MODELS.semantic.url).toContain(
      '/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite')
  })

  it('preserves mask dimensions, IDs and model labels and copies borrowed mask data', () => {
    const native = Uint8Array.of(0,12,12,15,0,12)
    const output=copySemanticCategoryMask({
      width:3,height:2,values:native,
      labels:['background','plane','bike'],
      qualityScores:[0.8,0.5],
    })
    expect(output.kind).toBe('semantic-segmentation')
    expect(output.width).toBe(3)
    expect(output.height).toBe(2)
    expect(Array.from(output.categoryIds)).toEqual([0,12,12,15,0,12])
    expect(output.labels).toEqual(['background','plane','bike'])
    expect(output.qualityScores).toEqual([.8,.5])
    native[1]=0
    expect(output.categoryIds[1]).toBe(12)
  })

  it('rejects malformed native masks instead of inventing segmentation', () => {
    expect(()=>copySemanticCategoryMask({width:2,height:2,
      values:Uint8Array.of(12),labels:[]})).toThrow(/length/)
    expect(()=>copySemanticCategoryMask({width:0,height:2,
      values:Uint8Array.of(0,0),labels:[]})).toThrow(/dimensions/)
    expect(()=>copySemanticCategoryMask({width:2,height:2,
      values:Uint8Array.of(0,1,2,3),labels:[],qualityScores:[NaN]})).toThrow(/quality/)
    expect(()=>copySemanticCategoryMask({width:2,height:2,
      values:Uint8Array.of(0,1,2,3),labels:['background',42 as never]})).toThrow(/label/)
  })

  it('protects against dangerous oversized mask allocations', () => {
    expect(()=>copySemanticCategoryMask({
      width:5000,height:5000,values:new Uint8Array(1),labels:[],
    })).toThrow(/dimensions/)
  })

  it('constructs cold and pre-aborted load never touches the network', async () => {
    const fetcher=vi.spyOn(globalThis,'fetch')
    try {
      const runner=new BrowserSubstrateRunner(createSemanticSegmenterSubstrate())
      expect(runner.status).toBe('idle')
      expect(runner.substrate.capabilities).toContain('image-segmentation')
      expect(fetcher).not.toHaveBeenCalled()
      const controller=new AbortController();controller.abort()
      await expect(runner.load(undefined,controller.signal))
        .rejects.toMatchObject({name:'AbortError'})
      expect(fetcher).not.toHaveBeenCalled()
    } finally {fetcher.mockRestore()}
  })
})
