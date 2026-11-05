import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS, createFaceDetectorSubstrate,
  mapFaceDetectorResult,
} from './index'

describe('MediaPipe BlazeFace short-range detector', () => {
  it('pins exact publisher bytes and full SHA-256', () => {
    expect(MEDIAPIPE_VISION_MODELS.blazeFace.bytes).toBe(229_746)
    expect(MEDIAPIPE_VISION_MODELS.blazeFace.sha256)
      .toBe('b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f')
    expect(MEDIAPIPE_VISION_MODELS.blazeFace.url)
      .toContain('face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite')
  })
  it('preserves pixel boxes, confidence and eye/nose/mouth/ear normalized keypoints', () => {
    const output=mapFaceDetectorResult({detections:[{
      boundingBox:{originX:55,originY:35,width:150,height:180,angle:0},
      categories:[{index:0,categoryName:'face',displayName:'Face',score:.93}],
      keypoints:[
        {x:.25,y:.35,label:'leftEye',score:.8},
        {x:.65,y:.35,label:'rightEye',score:.9},
      ],
    }]})
    expect(output.kind).toBe('face-detections')
    expect(output.detections).toHaveLength(1)
    expect(output.detections[0].boundingBox?.originX).toBe(55)
    expect(output.detections[0].categories[0].score).toBe(.93)
    expect(output.detections[0].keypoints.map(p=>p.label)).toEqual(['leftEye','rightEye'])
    expect(output.detections[0].keypoints[0].x).toBe(.25)
  })
  it('returns empty results without inventing faces or keypoints', () => {
    expect(mapFaceDetectorResult({detections:[]})).toEqual({kind:'face-detections',detections:[]})
  })
  it('rejects invalid native geometry and scores instead of silently discarding detections', () => {
    expect(()=>mapFaceDetectorResult({detections:[{
      boundingBox:{originX:0,originY:0,width:-1,height:30,angle:0},
      categories:[],keypoints:[],
    }]})).toThrow(/bounding box/)
    expect(()=>mapFaceDetectorResult({detections:[{
      categories:[{index:0,categoryName:'face',displayName:'',score:NaN}],keypoints:[],
    }]})).toThrow(/score/)
  })
  it('validates confidence and SDK delegate before any model fetch', () => {
    expect(()=>createFaceDetectorSubstrate({minDetectionConfidence:NaN})).toThrow(/minDetectionConfidence/)
    expect(()=>createFaceDetectorSubstrate({minSuppressionThreshold:1.4})).toThrow(/minSuppressionThreshold/)
    expect(()=>createFaceDetectorSubstrate({minDetectionConfidence:-.1})).toThrow(/minDetectionConfidence/)
    expect(()=>createFaceDetectorSubstrate({delegate:'WebGPU' as never})).toThrow(/delegate/)
    expect(()=>createFaceDetectorSubstrate({delegate:'GPU'})).not.toThrow()
    expect(()=>createFaceDetectorSubstrate({delegate:'CPU'})).not.toThrow()
  })
  it('is cold before explicit load, and aborts without fetching weights', async () => {
    const fetchSpy=vi.spyOn(globalThis,'fetch')
    try {
      const runner=new BrowserSubstrateRunner(createFaceDetectorSubstrate())
      expect(runner.status).toBe('idle')
      expect(runner.substrate.capabilities).toContain('image-detection')
      expect(fetchSpy).not.toHaveBeenCalled()
      const controller=new AbortController();controller.abort()
      await expect(runner.load(undefined,controller.signal))
        .rejects.toMatchObject({name:'AbortError'})
      expect(fetchSpy).not.toHaveBeenCalled()
    }finally{fetchSpy.mockRestore()}
  })
})
