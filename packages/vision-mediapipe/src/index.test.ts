import { describe, expect, it } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS,
  MEDIAPIPE_VISION_WASM_BASE,
  createFaceLandmarkerSubstrate,
  createGestureRecognizerSubstrate,
  mapFaceLandmarkerResult,
  mapGestureRecognizerResult,
  validateVisionFrame,
  verifyVisionModelBytes,
} from './index'

describe('MediaPipe Vision reusable substrate contract', () => {
  it('pins verified official model identities and a version-matched WASM runtime', () => {
    expect(MEDIAPIPE_VISION_WASM_BASE).toContain('@mediapipe/tasks-vision@1.2.2/wasm')
    expect(MEDIAPIPE_VISION_MODELS.face.bytes).toBe(3758596)
    expect(MEDIAPIPE_VISION_MODELS.face.sha256).toBe('64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff')
    expect(MEDIAPIPE_VISION_MODELS.gesture.bytes).toBe(8373440)
    expect(MEDIAPIPE_VISION_MODELS.gesture.sha256).toBe('97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482')
  })

  it('checks full content hash, not only byte length, and refuses unexpected sizes', async () => {
    const fixture = new Uint8Array([10, 11, 12, 13])
    const sha = await crypto.subtle.digest('SHA-256', fixture)
    const expected = Array.from(new Uint8Array(sha), v => v.toString(16).padStart(2,'0')).join('')
    await expect(verifyVisionModelBytes(fixture, {bytes: 4, sha256: expected})).resolves.toBeUndefined()
    await expect(verifyVisionModelBytes(fixture, {bytes: 5, sha256: expected})).rejects.toThrow(/size/i)
    await expect(verifyVisionModelBytes(new Uint8Array([10,11,12,14]), {bytes: 4, sha256: expected}))
      .rejects.toThrow(/SHA-256/i)
  })

  it('keeps every face individually paired with its own blendshape scores', () => {
    const output = mapFaceLandmarkerResult({
      faceLandmarks: [
        [{x: .1, y:.2, z:.3}],
        [{x: .4, y:.5, z:.6}],
      ],
      faceBlendshapes: [
        {categories:[{categoryName:'eyeBlinkLeft',score:.9, index:0, displayName:'blink'}]},
        {categories:[{categoryName:'jawOpen',score:.8, index:0, displayName:'jaw'}]},
      ],
    })
    expect(output.faces).toHaveLength(2)
    expect(output.faces[0].blendshapes[0]).toMatchObject({categoryName:'eyeBlinkLeft',score:.9})
    expect(output.faces[1].blendshapes[0]).toMatchObject({categoryName:'jawOpen',score:.8})
    expect(output.faces[1].landmarks[0]).toEqual({x:.4, y:.5, z:.6})
  })

  it('pairs gestures and handedness by hand index without inventing detection results', () => {
    const output = mapGestureRecognizerResult({
      gestures: [
        [{categoryName:'Thumb_Up',score:.99,index:0,displayName:''}],
        [{categoryName:'Victory',score:.8,index:0,displayName:''}],
      ],
      handedness: [
        [{categoryName:'Left',score:.9,index:0,displayName:''}],
        [{categoryName:'Right',score:.7,index:0,displayName:''}],
      ],
      landmarks: [
        [{x:.1,y:.2,z:.3}],
        [{x:.4,y:.5,z:.6}],
      ],
      worldLandmarks: [[],[]],
    })
    expect(output.hands).toHaveLength(2)
    expect(output.hands.map(h=>[h.topGesture?.categoryName,h.handedness])).toEqual([
      ['Thumb_Up','Left'], ['Victory','Right'],
    ])
    expect(mapGestureRecognizerResult({gestures:[],handedness:[],landmarks:[],worldLandmarks:[]}).hands).toEqual([])
  })

  it('rejects invalid video timestamps and repeated or backwards frames', () => {
    const image = {} as HTMLVideoElement
    expect(validateVisionFrame({image})).toBe('IMAGE')
    expect(validateVisionFrame({image,timestampMs:100})).toBe('VIDEO')
    expect(()=>validateVisionFrame({image,timestampMs:0},0)).toThrow(/increasing/i)
    expect(()=>validateVisionFrame({image,timestampMs:99},100)).toThrow(/increasing/i)
    expect(()=>validateVisionFrame({image,timestampMs:NaN})).toThrow(/timestamp/i)
    expect(()=>validateVisionFrame({image,timestampMs:-1})).toThrow(/timestamp/i)
    expect(()=>validateVisionFrame({image:null as unknown as HTMLVideoElement})).toThrow(/image/i)
  })

  it('constructs two cold, typed runners without beginning model downloads', () => {
    const face = new BrowserSubstrateRunner(createFaceLandmarkerSubstrate())
    const gesture = new BrowserSubstrateRunner(createGestureRecognizerSubstrate())
    expect([face.status,gesture.status]).toEqual(['idle','idle'])
    expect(face.substrate.capabilities).toContain('image-detection')
    expect(gesture.substrate.capabilities).toContain('image-pose')
  })

  it('does not attempt downloads when load is already aborted', async () => {
    const abort = new AbortController()
    abort.abort()
    const runner = new BrowserSubstrateRunner(createFaceLandmarkerSubstrate())
    await expect(runner.load(undefined,abort.signal)).rejects.toMatchObject({name:'AbortError'})
    expect(runner.status).toBe('idle')
  })
})
