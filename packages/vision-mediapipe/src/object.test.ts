import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS,
  createObjectDetectorSubstrate,
  mapObjectDetectorResult,
} from './index'

describe('MediaPipe Object Detector shared substrate', () => {
  it('pins official EfficientDet-Lite0 bytes and SHA-256', () => {
    expect(MEDIAPIPE_VISION_MODELS.object.bytes).toBe(13_836_895)
    expect(MEDIAPIPE_VISION_MODELS.object.sha256)
      .toBe('40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58')
    expect(MEDIAPIPE_VISION_MODELS.object.url)
      .toContain('object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite')
  })

  it('retains native pixel bounding boxes, per-object labels, exact scores and keypoints', () => {
    const result = mapObjectDetectorResult({
      detections: [
        {
          boundingBox: {originX:30,originY:40,width:260,height:190,angle:0},
          categories: [
            {index:17,categoryName:'dog',displayName:'Dog',score:0.943},
            {index:1,categoryName:'animal',displayName:'',score:0.2},
          ],
          keypoints: [{x:.25,y:.8,label:'nose',score:.77}],
        },
        {
          boundingBox: {originX:300,originY:10,width:100,height:90,angle:0},
          categories:[{index:15,categoryName:'cat',displayName:'Cat',score:.8}],
          keypoints:[],
        },
      ],
    })
    expect(result.kind).toBe('object-detections')
    expect(result.detections).toHaveLength(2)
    expect(result.detections[0].boundingBox).toEqual({
      originX:30,originY:40,width:260,height:190,angle:0,
    })
    expect(result.detections[0].categories.map(c=>[c.categoryName,c.score]))
      .toEqual([['dog',.943],['animal',.2]])
    expect(result.detections[0].keypoints[0]).toEqual({x:.25,y:.8,label:'nose',score:.77})
    expect(result.detections[1].categories[0].categoryName).toBe('cat')
  })

  it('does not invent bounding boxes or drop detections on empty/missing optional fields', () => {
    expect(mapObjectDetectorResult({detections:[]})).toEqual({kind:'object-detections',detections:[]})
    const mapped=mapObjectDetectorResult({
      detections:[{categories:[],keypoints:[]}],
    })
    expect(mapped.detections).toHaveLength(1)
    expect(mapped.detections[0].boundingBox).toBeUndefined()
  })

  it('rejects non-finite and impossible pixel geometry or classification scores', () => {
    expect(()=>mapObjectDetectorResult({detections:[{
      boundingBox:{originX:NaN,originY:0,width:1,height:1,angle:0},categories:[],keypoints:[],
    }]})).toThrow(/bounding box/)
    expect(()=>mapObjectDetectorResult({detections:[{
      boundingBox:{originX:0,originY:0,width:-1,height:1,angle:0},categories:[],keypoints:[],
    }]})).toThrow(/bounding box/)
    expect(()=>mapObjectDetectorResult({detections:[{
      categories:[{index:1,categoryName:'dog',displayName:'',score:Infinity}],keypoints:[],
    }]})).toThrow(/score/)
  })

  it('validates count and confidence options before loading a model', () => {
    expect(()=>createObjectDetectorSubstrate({maxResults:0})).toThrow(/maxResults/)
    expect(()=>createObjectDetectorSubstrate({maxResults:1000})).toThrow(/maxResults/)
    expect(()=>createObjectDetectorSubstrate({scoreThreshold:NaN})).toThrow(/scoreThreshold/)
    expect(()=>createObjectDetectorSubstrate({scoreThreshold:-.1})).toThrow(/scoreThreshold/)
    expect(()=>createObjectDetectorSubstrate({scoreThreshold:1.1})).toThrow(/scoreThreshold/)
  })

  it('constructs a cold runner and prevents pre-aborted fetch/init', async () => {
    const fetchSpy=vi.spyOn(globalThis,'fetch')
    try {
      const runner=new BrowserSubstrateRunner(createObjectDetectorSubstrate({maxResults:5}))
      expect(runner.status).toBe('idle')
      expect(runner.substrate.capabilities).toContain('image-detection')
      expect(fetchSpy).not.toHaveBeenCalled()
      const controller=new AbortController()
      controller.abort()
      await expect(runner.load(undefined,controller.signal))
        .rejects.toMatchObject({name:'AbortError'})
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      fetchSpy.mockRestore()
    }
  })
})
