import { describe, expect, it } from 'vitest'
import { BrowserSubstrateRunner, type BrowserSubstrate } from '@lil-local/inference-core'
import {
  AdaptiveSpeculationController,
  BoundedSessionTokenMemory,
  chooseSubmissionWindow,
  collectSuffixDraftCandidates,
} from '@lil-local/text-gen'
import { assertAddOutput } from './add-parity'
import { createUniversalRetrievalSubstrate, validateRetrievalRequest } from '@lil-local/retrieval-mediapipe'
import { createYamNetSubstrate, createParakeetSubstrate, createMediaPipeTtsSubstrate, mapYamNetResults } from '@lil-local/audio-mediapipe'
import { createFaceLandmarkerSubstrate, createGestureRecognizerSubstrate, createPoseLandmarkerSubstrate, createObjectDetectorSubstrate, createImageClassifierSubstrate, createHolisticLandmarkerSubstrate, createSemanticSegmenterSubstrate, createInteractiveSegmenterSubstrate, createFaceDetectorSubstrate, mapFaceDetectorResult, validateInteractiveStrokes, copyInteractiveMask, copySemanticCategoryMask, mapHolisticLandmarkerResult, mapImageClassifierResult, mapFaceLandmarkerResult, mapPoseLandmarkerResult, mapObjectDetectorResult } from '@lil-local/vision-mediapipe'

const fixture = (onDispose: () => void = () => {}) => ({
  id: 'external-packed-consumer',
  format: 'fixture',
  capabilities: ['text-generation'],
  requestedBackends: ['wasm'],
  async create() {
    return { infer: async (input: string) => input.toUpperCase(), dispose: async () => onDispose() }
  },
}) satisfies BrowserSubstrate<string, string>

describe('packed LiteRT numerical oracle', () => {
  it('checks every output against the requested input sum', () => {
    expect(assertAddOutput(Float32Array.of(8, 8), 3, 5)).toBe(0)
    expect(assertAddOutput(Float32Array.of(5, 5), -2, 7)).toBe(0)
    expect(() => assertAddOutput(Float32Array.of(8), -2, 7)).toThrow(/Numerical mismatch/)
    expect(() => assertAddOutput(Float32Array.of(NaN), 3, 5)).toThrow(/Non-finite/)
    expect(() => assertAddOutput(new Float32Array(), 3, 5)).toThrow(/Empty/)
  })
})

describe('installed packed text-gen speculation contract', () => {
  it('imports and executes the runtime-neutral speculation helpers', () => {
    expect(collectSuffixDraftCandidates(
      [1, 2, 3],
      [[9, 1, 2, 3, 4, 5]],
    )[0]).toEqual({
      tokens: [4, 5],
      matchOrder: 3,
      occurrences: 1,
    })

    const memory = new BoundedSessionTokenMemory(3)
    memory.append('model-a', [1, 2, 3, 4])
    expect(memory.snapshot('model-a')).toEqual([2, 3, 4])

    expect(chooseSubmissionWindow([
      { steps: 1, elapsedMs: 10 },
      { steps: 2, elapsedMs: 18 },
      { steps: 4, elapsedMs: 34 },
    ])).toBe(2)

    const controller = new AdaptiveSpeculationController([1, 2])
    controller.record({ width: 1, acceptedTokens: 1, elapsedMs: 4 })
    controller.record({ width: 2, acceptedTokens: 2, elapsedMs: 5 })
    expect(controller.selectPlan({ targetWindowMs: 10 })).toEqual({
      width: 2,
      windowSteps: 2,
    })
  })
})

describe('installed packed inference-core contract', () => {
  it('loads, executes, and disposes through the public entrypoint', async () => {
    let disposals = 0
    const runner = new BrowserSubstrateRunner(fixture(() => disposals++))
    expect(runner.status).toBe('idle')
    await runner.load()
    expect(runner.status).toBe('ready')
    expect(await runner.run('hello')).toBe('HELLO')
    await runner.dispose()
    expect(runner.status).toBe('disposed')
    expect(disposals).toBe(1)
  })

  it('does not construct the model when pre-aborted', async () => {
    let creates = 0
    const descriptor = { ...fixture(), create: async () => {
      creates++
      return { infer: async (input: string) => input, dispose: async () => {} }
    }}
    const runner = new BrowserSubstrateRunner(descriptor)
    const controller = new AbortController()
    controller.abort()
    await expect(runner.load(undefined, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(creates).toBe(0)
    expect(runner.status).toBe('idle')
  })

  it('propagates inference failures without swapping in a CPU result', async () => {
    const runner = new BrowserSubstrateRunner({
      ...fixture(),
      create: async () => ({ infer: async (_input: string): Promise<string> => { throw new Error('GPU failed') }, dispose: async () => {} }),
    })
    await runner.load()
    await expect(runner.run('hello')).rejects.toThrow('GPU failed')
    expect(runner.status).toBe('ready')
    await runner.dispose()
  })

  it('waits for active inference before releasing packed-model resources', async () => {
    let finish!: (value: string) => void
    const pending = new Promise<string>((resolve) => { finish = resolve })
    let released = 0
    const runner = new BrowserSubstrateRunner({
      ...fixture(),
      create: async () => ({
        infer: (_input: string) => pending,
        dispose: async () => { released++ },
      }),
    })
    await runner.load()
    const result = runner.run('input')
    const closing = runner.dispose()
    expect(released).toBe(0)
    await expect(runner.run('again')).rejects.toThrow(/Cannot infer/)
    finish('done')
    await expect(result).resolves.toBe('done')
    await closing
    expect(released).toBe(1)
    expect(runner.status).toBe('disposed')
  })

  it('disposes a model returned after its load was aborted', async () => {
    let finishLoad!: (instance: { infer(input: string): Promise<string>; dispose(): Promise<void> }) => void
    const pending = new Promise<{ infer(input: string): Promise<string>; dispose(): Promise<void> }>(
      (resolve) => { finishLoad = resolve },
    )
    let released = 0
    const runner = new BrowserSubstrateRunner({ ...fixture(), create: async () => pending })
    const controller = new AbortController()
    const loading = runner.load(undefined, controller.signal)
    controller.abort()
    finishLoad({ infer: async (input) => input, dispose: async () => { released++ } })
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(released).toBe(1)
    expect(runner.status).toBe('idle')
  })
})


describe('installed packed vision-mediapipe public contract', () => {
  it('instantiates cold runners and maps distinct face results without starting a model download', () => {
    const face = new BrowserSubstrateRunner(createFaceLandmarkerSubstrate())
    const gesture = new BrowserSubstrateRunner(createGestureRecognizerSubstrate())
    const pose = new BrowserSubstrateRunner(createPoseLandmarkerSubstrate({numPoses:2}))
    const object = new BrowserSubstrateRunner(createObjectDetectorSubstrate({maxResults:3}))
    const classifier = new BrowserSubstrateRunner(createImageClassifierSubstrate({maxResults:3}))
    const holistic = new BrowserSubstrateRunner(createHolisticLandmarkerSubstrate())
    const semantic = new BrowserSubstrateRunner(createSemanticSegmenterSubstrate())
    const interactive = new BrowserSubstrateRunner(createInteractiveSegmenterSubstrate())
    const blazeFace = new BrowserSubstrateRunner(createFaceDetectorSubstrate({delegate:'GPU'}))
    expect([face.status, gesture.status, pose.status, object.status, classifier.status, holistic.status, semantic.status, interactive.status, blazeFace.status]).toEqual(['idle','idle','idle','idle','idle','idle','idle','idle','idle'])
    expect(mapFaceDetectorResult({detections:[{categories:[],keypoints:[]}]}).detections).toHaveLength(1)
    expect(interactive.substrate.capabilities).toContain('image-segmentation')
    expect(validateInteractiveStrokes([{mode:'positive',points:[{x:.1,y:.2},{x:.3,y:.4}]}])[0].brushMode).toBe(1)
    expect(copyInteractiveMask({width:1,height:1,values:Float32Array.of(.9)}).confidence[0]).toBeCloseTo(.9)
    expect(copySemanticCategoryMask({width:2,height:1,values:Uint8Array.of(0,12),labels:[]}).categoryIds[1]).toBe(12)
    expect(holistic.substrate.capabilities).toContain('holistic-landmarks')
    expect(mapHolisticLandmarkerResult({faceLandmarks:[],faceBlendshapes:[],poseLandmarks:[],poseWorldLandmarks:[],leftHandLandmarks:[],leftHandWorldLandmarks:[],rightHandLandmarks:[],rightHandWorldLandmarks:[]}).hands).toEqual([])
    expect(mapImageClassifierResult({classifications:[{headIndex:0,headName:'main',categories:[{index:1,categoryName:'dog',displayName:'Dog',score:.75}]}]}).heads[0].categories[0].score).toBe(.75)
    expect(mapObjectDetectorResult({detections:[{categories:[],keypoints:[]}]}).detections).toHaveLength(1)
    expect(mapPoseLandmarkerResult({landmarks:[[{x:.1,y:.2,z:.3}]],worldLandmarks:[[{x:1,y:2,z:3}]]}).poses[0].worldLandmarks[0].z).toBe(3)
    const mapped = mapFaceLandmarkerResult({
      faceLandmarks: [[{x:.2,y:.3,z:.4}]],
      faceBlendshapes: [{categories:[{categoryName:'jawOpen',score:.8}]}],
    })
    expect(mapped.faces[0].blendshapes[0].score).toBe(.8)
  })
})


describe('installed packed MediaPipe audio substrate', () => {
  it('instantiates all three opt-in model tasks without network access', () => {
    const yamnet = new BrowserSubstrateRunner(createYamNetSubstrate())
    const parakeet = new BrowserSubstrateRunner(createParakeetSubstrate())
    const tts = new BrowserSubstrateRunner(createMediaPipeTtsSubstrate({model:new Uint8Array([1,2,3])}))
    expect([yamnet.status,parakeet.status,tts.status]).toEqual(['idle','idle','idle'])
    expect(mapYamNetResults([]).windows).toEqual([])
    expect(yamnet.substrate.capabilities).toContain('audio-classification')
  })
})


describe('installed packed MediaPipe universal retrieval surface', () => {
  it('constructs the typed universal embedding + semantic retriever without loading model weights', () => {
    const runner = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
      model: new Blob(['caller-owned fixture']),
    }))
    expect(runner.status).toBe('idle')
    expect(runner.substrate.capabilities).toContain('semantic-retrieval')
    expect(runner.substrate.capabilities).toContain('image-embedding')
    expect(() => validateRetrievalRequest({kind:'search',query:'local photo',limit:5})).not.toThrow()
  })
})
