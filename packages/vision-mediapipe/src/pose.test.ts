import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS,
  createPoseLandmarkerSubstrate,
  mapPoseLandmarkerResult,
} from './index'

describe('MediaPipe Pose Landmarker downstream contract', () => {
  it('pins actual published lite weights by size and digest', () => {
    expect(MEDIAPIPE_VISION_MODELS.pose.bytes).toBe(5_777_746)
    expect(MEDIAPIPE_VISION_MODELS.pose.sha256)
      .toBe('59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a')
    expect(MEDIAPIPE_VISION_MODELS.pose.url).toContain('pose_landmarker_lite/float16/1/')
  })

  it('preserves pose-index pairs, normalized and metric world landmarks, and presence metadata', () => {
    const output=mapPoseLandmarkerResult({
      landmarks: [
        [{x:.1,y:.2,z:.3,visibility:.8,presence:.9}],
        [{x:.4,y:.5,z:.6,visibility:.4,presence:.3}],
      ],
      worldLandmarks: [
        [{x:1,y:2,z:3,visibility:.7,presence:.6}],
        [{x:4,y:5,z:6,visibility:.5,presence:.2}],
      ],
    })
    expect(output.kind).toBe('pose-landmarks')
    expect(output.poses).toHaveLength(2)
    expect(output.poses[0].landmarks[0]).toEqual({x:.1,y:.2,z:.3,visibility:.8,presence:.9})
    expect(output.poses[1].worldLandmarks[0]).toEqual({x:4,y:5,z:6,visibility:.5,presence:.2})
    expect(output.poses[1].landmarks[0].x).toBe(.4)
    expect(mapPoseLandmarkerResult({landmarks:[],worldLandmarks:[]}).poses).toEqual([])
  })

  it('does not fabricate missing visibility/presence or conflate world coords with normalized', () => {
    const output=mapPoseLandmarkerResult({
      landmarks:[[{x:.1,y:.2,z:.3}]],
      worldLandmarks:[[{x:10,y:20,z:30}]],
    })
    expect(output.poses[0].landmarks[0]).toEqual({x:.1,y:.2,z:.3})
    expect(output.poses[0].worldLandmarks[0]).toEqual({x:10,y:20,z:30})
    expect(output.poses[0].worldLandmarks[0].x).not.toBe(output.poses[0].landmarks[0].x)
  })

  it('rejects inconsistent indexed pose arrays rather than silently matching wrong bodies', () => {
    expect(()=>mapPoseLandmarkerResult({
      landmarks:[[{x:1,y:2,z:3}],[{x:4,y:5,z:6}]],
      worldLandmarks:[[{x:1,y:2,z:3}]],
    })).toThrow(/pose.*count/i)
  })

  it('validates number of people and confidence thresholds before any download', () => {
    expect(()=>createPoseLandmarkerSubstrate({numPoses:0})).toThrow(/numPoses/)
    expect(()=>createPoseLandmarkerSubstrate({numPoses:1.5})).toThrow(/numPoses/)
    expect(()=>createPoseLandmarkerSubstrate({minPoseDetectionConfidence:-.1})).toThrow(/Confidence/)
    expect(()=>createPoseLandmarkerSubstrate({minTrackingConfidence:1.1})).toThrow(/Confidence/)
    expect(()=>createPoseLandmarkerSubstrate({minPosePresenceConfidence:NaN})).toThrow(/Confidence/)
  })

  it('stays cold until explicitly loaded and refuses pre-aborted load', async () => {
    const fetcher=vi.spyOn(globalThis,'fetch')
    const runner=new BrowserSubstrateRunner(createPoseLandmarkerSubstrate({numPoses:2}))
    expect(runner.status).toBe('idle')
    expect(runner.substrate.capabilities).toContain('image-pose')
    expect(fetcher).not.toHaveBeenCalled()
    const abort=new AbortController();abort.abort()
    await expect(runner.load(undefined,abort.signal)).rejects.toMatchObject({name:'AbortError'})
    expect(fetcher).not.toHaveBeenCalled()
    fetcher.mockRestore()
  })
})
