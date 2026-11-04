import type { BrowserSubstrate } from '@lil-local/inference-core'
import type { FaceLandmarker, GestureRecognizer } from '@mediapipe/tasks-vision'

export const MEDIAPIPE_VISION_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm'

/** Exact bytes and SHA-256 verified against the publisher's downloadable assets (2026-11-04). */
export const MEDIAPIPE_VISION_MODELS = {
  face: {
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    bytes: 3_758_596,
    sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
  },
  gesture: {
    url: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
    bytes: 8_373_440,
    sha256: '97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482',
  },
} as const

export type VisionKind = keyof typeof MEDIAPIPE_VISION_MODELS
export type VisionImage = ImageData | HTMLCanvasElement | HTMLImageElement | HTMLVideoElement
export interface VisionFrame {
  image: VisionImage
  /** Provide for video frames; must be strictly increasing per streaming run. */
  timestampMs?: number
}
export type VisionProgress = 'model' | 'integrity' | 'wasm'
export interface VisionOptions {
  /** Use caller-owned official model bytes to avoid a model fetch; integrity verification is still required. */
  model?: Blob | Uint8Array
  /** Self-host the exact matching WASM files for offline operation. */
  wasmBase?: string
}
export interface FaceVisionOptions extends VisionOptions { numFaces?: number }
export interface GestureVisionOptions extends VisionOptions { numHands?: number }
export interface VisionLandmark { x: number; y: number; z: number }
export interface VisionScore { categoryName: string; score: number }
export interface FaceVisionResult {
  kind: 'face-landmarks'
  faces: { landmarks: VisionLandmark[]; blendshapes: VisionScore[] }[]
}
export interface GestureVisionResult {
  kind: 'gestures'
  hands: {
    landmarks: VisionLandmark[]
    worldLandmarks: VisionLandmark[]
    gestures: VisionScore[]
    topGesture: VisionScore | null
    handedness: 'Left' | 'Right' | 'Unknown'
  }[]
}

type MinimalLandmark = { x: number; y: number; z?: number }
type MinimalCategory = { categoryName: string; score: number; index?: number; displayName?: string }
const convertLandmarks = (points: readonly MinimalLandmark[] = []): VisionLandmark[] =>
  points.map(({ x, y, z }) => ({ x, y, z: z ?? 0 }))
const convertCategories = (categories: readonly MinimalCategory[] = []): VisionScore[] =>
  categories.filter(x => Number.isFinite(x.score)).map(({ categoryName, score }) => ({ categoryName, score }))

/** Preserve per-face alignment. Never use first face's blendshapes for every face. */
export function mapFaceLandmarkerResult(
  result: { faceLandmarks: readonly (readonly MinimalLandmark[])[]; faceBlendshapes: readonly { categories: readonly MinimalCategory[] }[] },
): FaceVisionResult {
  return {
    kind: 'face-landmarks',
    faces: result.faceLandmarks.map((landmarks, i) => ({
      landmarks: convertLandmarks(landmarks),
      blendshapes: convertCategories(result.faceBlendshapes[i]?.categories),
    })),
  }
}

/** Keep all real scores; downstream apps decide gesture actions and safety gating. */
export function mapGestureRecognizerResult(
  result: { gestures: readonly (readonly MinimalCategory[])[]; handedness: readonly (readonly MinimalCategory[])[]; landmarks: readonly (readonly MinimalLandmark[])[]; worldLandmarks: readonly (readonly MinimalLandmark[])[] },
): GestureVisionResult {
  return {
    kind: 'gestures',
    hands: result.gestures.map((classifications, i) => {
      const gestures = convertCategories(classifications)
      const label = result.handedness[i]?.[0]?.categoryName
      return {
        landmarks: convertLandmarks(result.landmarks[i]),
        worldLandmarks: convertLandmarks(result.worldLandmarks[i]),
        gestures,
        topGesture: gestures[0] ?? null,
        handedness: label === 'Left' || label === 'Right' ? label : 'Unknown',
      }
    }),
  }
}

export function validateVisionFrame(frame: VisionFrame, lastVideoTimestamp = -Infinity): 'IMAGE' | 'VIDEO' {
  if (!frame?.image) throw new Error('Vision image is required')
  if (frame.timestampMs === undefined) return 'IMAGE'
  if (!Number.isFinite(frame.timestampMs) || frame.timestampMs < 0)
    throw new Error('Vision timestamp must be a finite nonnegative number of milliseconds')
  if (frame.timestampMs <= lastVideoTimestamp)
    throw new Error('Vision video timestamps must be strictly increasing')
  return 'VIDEO'
}

export async function verifyVisionModelBytes(
  bytes: Uint8Array,
  expected: { bytes: number; sha256: string },
): Promise<void> {
  if (bytes.byteLength !== expected.bytes)
    throw new Error('MediaPipe model size mismatch')
  if (!globalThis.crypto?.subtle)
    throw new Error('MediaPipe model SHA-256 requires secure WebCrypto context')
  // Copy to an owned ArrayBuffer to satisfy newer TypeScript BufferSource variance.
  const digest = await globalThis.crypto.subtle.digest('SHA-256', Uint8Array.from(bytes))
  const actual = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('')
  if (actual !== expected.sha256) throw new Error('MediaPipe model SHA-256 mismatch')
}

function aborted(): Error {
  const error = new Error('MediaPipe Vision loading aborted')
  error.name = 'AbortError'
  return error
}
function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw aborted()
}

async function modelBytes(
  kind: VisionKind,
  options: VisionOptions,
  progress?: (value: VisionProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  checkAbort(signal)
  progress?.('model')
  let bytes: Uint8Array
  if (options.model instanceof Uint8Array) {
    bytes = options.model
  } else if (options.model instanceof Blob) {
    bytes = new Uint8Array(await options.model.arrayBuffer())
  } else {
    const manifest = MEDIAPIPE_VISION_MODELS[kind]
    const response = await fetch(manifest.url, { signal })
    if (!response.ok) throw new Error('MediaPipe Vision model download failed: HTTP ' + response.status)
    const header = response.headers.get('content-length')
    if (header && Number(header) !== manifest.bytes)
      throw new Error('MediaPipe model size mismatch in HTTP response')
    bytes = new Uint8Array(await response.arrayBuffer())
  }
  checkAbort(signal)
  progress?.('integrity')
  await verifyVisionModelBytes(bytes, MEDIAPIPE_VISION_MODELS[kind])
  checkAbort(signal)
  return bytes
}

function assertCount(count: number, field: string): void {
  if (!Number.isInteger(count) || count < 1 || count > 8)
    throw new Error(field + ' must be an integer from 1 to 8')
}

/**
 * Passive descriptor until BrowserSubstrateRunner.load(). No webcam is opened
 * here; callers own camera permissions, frames, scheduling, and OBS actions.
 */
export function createFaceLandmarkerSubstrate(
  options: FaceVisionOptions = {},
): BrowserSubstrate<VisionFrame, FaceVisionResult, VisionProgress> {
  const count = options.numFaces ?? 1
  assertCount(count, 'numFaces')
  return {
    id: 'mediapipe:face-landmarker-float16-1',
    format: 'mediapipe-task',
    capabilities: ['image-detection'],
    requestedBackends: ['wasm'],
    async create(progress, signal) {
      const bytes = await modelBytes('face', options, progress, signal)
      progress?.('wasm')
      const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision')
      checkAbort(signal)
      const fileset = await FilesetResolver.forVisionTasks(options.wasmBase ?? MEDIAPIPE_VISION_WASM_BASE)
      checkAbort(signal)
      let task: FaceLandmarker | null = null
      try {
        task = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: bytes, delegate: 'CPU' },
          runningMode: 'IMAGE',
          numFaces: count,
          outputFaceBlendshapes: true,
        })
        checkAbort(signal)
        const engine = task
        let mode: 'IMAGE' | 'VIDEO' = 'IMAGE'
        let lastVideoTimestamp = -Infinity
        task = null
        return {
          async infer(frame) {
            const nextMode = validateVisionFrame(frame, mode === 'VIDEO' ? lastVideoTimestamp : -Infinity)
            if (nextMode !== mode) {
              await engine.setOptions({ runningMode: nextMode })
              mode = nextMode
              if (mode === 'IMAGE') lastVideoTimestamp = -Infinity
            }
            if (nextMode === 'VIDEO') {
              const timestamp = frame.timestampMs!
              const result = engine.detectForVideo(frame.image, timestamp)
              lastVideoTimestamp = timestamp
              return mapFaceLandmarkerResult(result)
            }
            return mapFaceLandmarkerResult(engine.detect(frame.image))
          },
          async dispose() { engine.close() },
        }
      } finally {
        task?.close()
      }
    },
  }
}

export function createGestureRecognizerSubstrate(
  options: GestureVisionOptions = {},
): BrowserSubstrate<VisionFrame, GestureVisionResult, VisionProgress> {
  const count = options.numHands ?? 1
  assertCount(count, 'numHands')
  return {
    id: 'mediapipe:gesture-recognizer-float16-1',
    format: 'mediapipe-task',
    capabilities: ['image-pose'],
    requestedBackends: ['wasm'],
    async create(progress, signal) {
      const bytes = await modelBytes('gesture', options, progress, signal)
      progress?.('wasm')
      const { GestureRecognizer, FilesetResolver } = await import('@mediapipe/tasks-vision')
      checkAbort(signal)
      const fileset = await FilesetResolver.forVisionTasks(options.wasmBase ?? MEDIAPIPE_VISION_WASM_BASE)
      checkAbort(signal)
      let task: GestureRecognizer | null = null
      try {
        task = await GestureRecognizer.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: bytes, delegate: 'CPU' },
          runningMode: 'IMAGE',
          numHands: count,
        })
        checkAbort(signal)
        const engine = task
        let mode: 'IMAGE' | 'VIDEO' = 'IMAGE'
        let lastVideoTimestamp = -Infinity
        task = null
        return {
          async infer(frame) {
            const nextMode = validateVisionFrame(frame, mode === 'VIDEO' ? lastVideoTimestamp : -Infinity)
            if (nextMode !== mode) {
              await engine.setOptions({ runningMode: nextMode })
              mode = nextMode
              if (mode === 'IMAGE') lastVideoTimestamp = -Infinity
            }
            if (nextMode === 'VIDEO') {
              const timestamp = frame.timestampMs!
              const result = engine.recognizeForVideo(frame.image, timestamp)
              lastVideoTimestamp = timestamp
              return mapGestureRecognizerResult(result)
            }
            return mapGestureRecognizerResult(engine.recognize(frame.image))
          },
          async dispose() { engine.close() },
        }
      } finally {
        task?.close()
      }
    },
  }
}
