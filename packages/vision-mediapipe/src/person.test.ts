import {describe,expect,it,vi} from 'vitest'
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {MEDIAPIPE_VISION_MODELS,createPersonSegmenterSubstrate,copyPersonMask} from './index'

describe('ControlNet-style MediaPipe selfie foreground mask',()=>{
 it('pins the publisher-verified model rather than confusing it with semantic segmentation',()=>{
  expect(MEDIAPIPE_VISION_MODELS.person).toEqual({
   url:'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite',
   bytes:249537,
   sha256:'191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b',
  })
 })
 it('copies the foreground confidence before MediaPipe releases native memory',()=>{
  const borrowed=Float32Array.from([0,0.25,0.5,1])
  const result=copyPersonMask({width:2,height:2,values:borrowed})
  expect(result.kind).toBe('person-mask')
  expect(Array.from(result.confidence)).toEqual([0,0.25,0.5,1])
  borrowed[1]=0
  expect(result.confidence[1]).toBe(.25)
 })
 it('rejects malformed dimensions and non-finite or out-of-range probabilities',()=>{
  expect(()=>copyPersonMask({width:2,height:2,values:Float32Array.of(0)})).toThrow(/length/)
  expect(()=>copyPersonMask({width:0,height:2,values:Float32Array.of(0)})).toThrow(/dimension/)
  expect(()=>copyPersonMask({width:1,height:1,values:Float32Array.of(NaN)})).toThrow(/probability/)
  expect(()=>copyPersonMask({width:1,height:1,values:Float32Array.of(1.2)})).toThrow(/probability/)
 })
 it('does not fetch until explicit load and honors pre-abort',async()=>{
  const fetcher=vi.spyOn(globalThis,'fetch')
  try{
   const runner=new BrowserSubstrateRunner(createPersonSegmenterSubstrate())
   expect(runner.substrate.capabilities).toContain('image-segmentation')
   expect(fetcher).not.toHaveBeenCalled()
   const aborted=new AbortController();aborted.abort()
   await expect(runner.load(undefined,aborted.signal)).rejects.toMatchObject({name:'AbortError'})
   expect(fetcher).not.toHaveBeenCalled()
  }finally{fetcher.mockRestore()}
 })
})
