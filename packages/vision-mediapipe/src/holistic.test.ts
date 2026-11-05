import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS, createHolisticLandmarkerSubstrate,
  mapHolisticLandmarkerResult,
} from './index'

const lm=(x:number,y:number,z:number,visibility?:number,presence?:number)=>({
  x,y,z,...(visibility===undefined?{}:{visibility}),...(presence===undefined?{}:{presence}),
})
const empty=()=>({
  faceLandmarks: [],faceBlendshapes:[],poseLandmarks:[],poseWorldLandmarks:[],
  leftHandLandmarks:[],leftHandWorldLandmarks:[],
  rightHandLandmarks:[],rightHandWorldLandmarks:[],
})

describe('MediaPipe holistic single-model structured body landmarks',()=>{
  it('pins real official model bytes and SHA-256 without mutable latest',()=>{
    expect(MEDIAPIPE_VISION_MODELS.holistic.url)
      .toContain('holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task')
    expect(MEDIAPIPE_VISION_MODELS.holistic.bytes).toBe(13_683_609)
    expect(MEDIAPIPE_VISION_MODELS.holistic.sha256)
      .toBe('e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8')
  })
  it('keeps face, pose-world, explicit left/right hands, native scores distinct',()=>{
    const result=mapHolisticLandmarkerResult({
      faceLandmarks:[[lm(.1,.2,.3)]],
      faceBlendshapes:[{categories:[{index:10,categoryName:'jawOpen',displayName:'Jaw',score:.82}]}],
      poseLandmarks:[[lm(.4,.5,.6,.95,.9)]],
      poseWorldLandmarks:[[lm(4,5,6,.8,.7)]],
      leftHandLandmarks:[[lm(.11,.12,.13)]],
      leftHandWorldLandmarks:[[lm(1,2,3)]],
      rightHandLandmarks:[[lm(.9,.8,.7)]],
      rightHandWorldLandmarks:[[lm(9,8,7)]],
    })
    expect(result.kind).toBe('holistic-landmarks')
    expect(result.faces[0].landmarks[0]).toEqual(lm(.1,.2,.3))
    expect(result.faces[0].blendshapes).toEqual([{categoryName:'jawOpen',score:.82}])
    expect(result.poses[0].landmarks[0].visibility).toBe(.95)
    expect(result.poses[0].worldLandmarks[0]).toEqual(lm(4,5,6,.8,.7))
    expect(result.hands.map(h=>h.side)).toEqual(['left','right'])
    expect(result.hands[0].worldLandmarks[0].z).toBe(3)
    expect(result.hands[1].worldLandmarks[0].z).toBe(7)
    expect(result.hands[0]).not.toHaveProperty('gestures')
  })
  it('returns honest empty detections for blank images, no fabricated landmarks',()=>{
    expect(mapHolisticLandmarkerResult(empty())).toEqual({
      kind:'holistic-landmarks',faces:[],poses:[],hands:[],
    })
  })
  it('never pairs unrelated native face/pose/hand arrays when counts disagree',()=>{
    const source=empty()
    expect(()=>mapHolisticLandmarkerResult({...source,
      poseLandmarks:[[lm(1,2,3)]],poseWorldLandmarks:[],
    })).toThrow(/pose.*count/i)
    expect(()=>mapHolisticLandmarkerResult({...source,
      leftHandLandmarks:[[lm(1,2,3)]],leftHandWorldLandmarks:[],
    })).toThrow(/left.*count/i)
    expect(()=>mapHolisticLandmarkerResult({...source,
      faceLandmarks:[[lm(1,2,3)]],faceBlendshapes:[{categories:[]},{categories:[]}],
    })).toThrow(/face.*count/i)
  })
  it('validates scores and landmark geometry without silently replacing invalid readings',()=>{
    const source=empty()
    expect(()=>mapHolisticLandmarkerResult({...source,
      faceLandmarks:[[lm(NaN,0,0)]],faceBlendshapes:[],
    })).toThrow(/finite/)
    expect(()=>mapHolisticLandmarkerResult({...source,
      faceLandmarks:[[lm(.1,.2,.3)]],
      faceBlendshapes:[{categories:[{categoryName:'jawOpen',score:NaN}]}],
    })).toThrow(/score/)
    expect(()=>mapHolisticLandmarkerResult({...source,
      poseLandmarks:[[lm(.1,.2,.3,1.1)]],poseWorldLandmarks:[[lm(0,0,0)]],
    })).toThrow(/visibility/)
  })
  it('validates confidence inputs and retains explicit zero threshold',()=>{
    expect(()=>createHolisticLandmarkerSubstrate({minPoseDetectionConfidence:NaN})).toThrow(/Confidence/)
    expect(()=>createHolisticLandmarkerSubstrate({minFaceDetectionConfidence:-.01})).toThrow(/Confidence/)
    expect(()=>createHolisticLandmarkerSubstrate({minHandLandmarksConfidence:1.2})).toThrow(/Confidence/)
    expect(()=>createHolisticLandmarkerSubstrate({minFacePresenceConfidence:0})).not.toThrow()
  })
  it('is cold on construction, advertises holistic capability, and pre-aborts without fetch',async()=>{
    const spy=vi.spyOn(globalThis,'fetch')
    try{
      const runner=new BrowserSubstrateRunner(createHolisticLandmarkerSubstrate())
      expect(runner.status).toBe('idle')
      expect(runner.substrate.capabilities).toContain('holistic-landmarks')
      expect(spy).not.toHaveBeenCalled()
      const controller=new AbortController();controller.abort()
      await expect(runner.load(undefined,controller.signal))
        .rejects.toMatchObject({name:'AbortError'})
      expect(spy).not.toHaveBeenCalled()
    }finally{spy.mockRestore()}
  })
})
