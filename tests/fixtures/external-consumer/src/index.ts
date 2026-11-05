import { BrowserSubstrateRunner, InferenceError, createHttpAssetResolver, type BrowserSubstrate } from '@lil-local/inference-core'
import { createLiteRtRuntime } from '@lil-local/runtime-litert'
import { ManagedLiteRtModelController, useManagedLiteRtModel } from '@lil-local/react-litert'
import { createBrowserCacheAssetResolver } from '@lil-local/browser-cache'
import {
  AdaptiveSpeculationController,
  BoundedSessionTokenMemory,
  LiteRtLmTextPipeline,
  LiteRtLmWorkerClient,
  chooseSubmissionWindow,
  collectSuffixDraftCandidates,
  type LiteRtLmTextPipelineOptions,
  type LiteRtLmWorkerLoadOptions,
  type LiteRtLmWorkerPrompt,
} from '@lil-local/text-gen'
import { createEmbeddingGemma2Substrate, formatEmbeddingGemma2Text } from '@lil-local/embedding-litertlm'
import { createUniversalRetrievalSubstrate, type UniversalRetrievalInput } from '@lil-local/retrieval-mediapipe'
import { EncoderPipeline } from '@lil-local/encoder'
import { createPrivacyFilterSubstrate, redactPrivacySpans, type PrivacyResult } from '@lil-local/privacy-filter'
import { D1OnnxWebGpu, type D1Question } from '@lil-local/decision-onnx'
import { SharpOnnxWebGpu, buildSharpPly, decodeSharpGaussians, type SharpRunOutput } from '@lil-local/sharp-onnx'
import {
  createPix2PixEdge2ShoesSubstrate,
  pix2pixEdge2ShoesManifest,
} from '@lil-local/pix2pix-webgpu'
import { ColBertPipeline } from '@lil-local/retrieval'
import { maxSim, rankDense } from '@lil-local/retrieval/scoring'
import { KokoroPipeline } from '@lil-local/kokoro'
import { PhononPipeline } from '@lil-local/phonon'
import { Qwen3TtsPipeline } from '@lil-local/qwen3-tts'
import { ClipImageEmbeddingPipeline } from '@lil-local/image-embedding'
import { MoViNetPipeline } from '@lil-local/video-classification'
import { createYamNetSubstrate, createParakeetSubstrate, createMediaPipeTtsSubstrate, YAMNET_MODEL, type AudioClip } from '@lil-local/audio-mediapipe'
import { createFaceLandmarkerSubstrate, createGestureRecognizerSubstrate, createPoseLandmarkerSubstrate, createObjectDetectorSubstrate, createImageClassifierSubstrate, createHolisticLandmarkerSubstrate, createSemanticSegmenterSubstrate, createInteractiveSegmenterSubstrate, MEDIAPIPE_VISION_MODELS, type VisionFrame, type FaceVisionResult, type GestureVisionResult, type PoseVisionResult, type ObjectVisionResult, type ImageClassifierVisionResult, type HolisticVisionResult, type SemanticVisionResult, type InteractiveSelectionRequest, type InteractiveSelectionResult } from '@lil-local/vision-mediapipe'
import {
  DepthEstimationPipeline,
  da3SmallPortraitManifest,
  estimateMoGeGeometry,
  releaseMoGeModels,
  moGeGeometryToDepthFrame,
  MOGE_REVISION,
  type MoGeGeometry,
  type DepthFrame,
} from '@lil-local/depth-estimation'

import {
  FACE_PARSING_MODEL, SAM2_MODEL_VARIANTS, suggestFaceRegions,
  selectSamPoint, selectSamPoints, prepareSamPointPrompts, validateRegion,
  SaliencySegmentationPipeline, u2NetSaliencyManifest, extractMaskLayer,
  type SegmentationRegion, type Sam2Variant, type SamPointPrompt, type ForegroundMask,
} from '@lil-local/image-segmentation'

import {
  createModerationSubstrate,
  MODERATION_CATEGORIES,
  MODERATION_MODEL_ID,
  MODERATION_MODEL_REVISION,
  type ModerationResult,
} from '@lil-local/moderation'

// Packed downstream consumers can instantiate the same reusable vision substrate
// without installing SDK internals or accessing a Lil Local workspace source path.
export const packedFaceRunner = new BrowserSubstrateRunner(createFaceLandmarkerSubstrate())
export const packedGestureRunner = new BrowserSubstrateRunner(createGestureRecognizerSubstrate())
export const packedPoseRunner = new BrowserSubstrateRunner(createPoseLandmarkerSubstrate({numPoses:2}))
export const packedObjectRunner = new BrowserSubstrateRunner(createObjectDetectorSubstrate({maxResults:5}))
export const packedClassifierRunner = new BrowserSubstrateRunner(createImageClassifierSubstrate({maxResults:5}))
export const packedHolisticRunner = new BrowserSubstrateRunner(createHolisticLandmarkerSubstrate())
export const packedSemanticRunner = new BrowserSubstrateRunner(createSemanticSegmenterSubstrate())
export const packedInteractiveRunner = new BrowserSubstrateRunner(createInteractiveSegmenterSubstrate())
export type PackedInteractiveRequest = InteractiveSelectionRequest
export type PackedInteractiveResult = InteractiveSelectionResult
export type PackedSemanticVision = SemanticVisionResult
export type PackedHolisticVision = HolisticVisionResult
export type PackedImageClassifierVision = ImageClassifierVisionResult
export type PackedObjectVision = ObjectVisionResult
export type PackedPoseVision = PoseVisionResult
export type PackedVisionFrame = VisionFrame
export type PackedFaceVision = FaceVisionResult
export type PackedGestureVision = GestureVisionResult
export const packedFaceModelSha256: string = MEDIAPIPE_VISION_MODELS.face.sha256

// All three MediaPipe Audio tasks resolve from the packed public entrypoint.
export const packedYamNetRunner = new BrowserSubstrateRunner(createYamNetSubstrate())
export const packedParakeetRunner = new BrowserSubstrateRunner(createParakeetSubstrate())
export const packedTtsRunner = new BrowserSubstrateRunner(createMediaPipeTtsSubstrate({
  model: new Blob(['placeholder model; never loaded during discovery']),
}))
export const packedYamNetSha256: string = YAMNET_MODEL.sha256
export type PackedAudioClip = AudioClip

// Real downstream packed MediaPipe universal embedder + SemanticRetriever,
// sharing exactly one caller-owned model and native provider.
export const packedUniversalRetriever = new BrowserSubstrateRunner(createUniversalRetrievalSubstrate({
  model: new Blob(['cold fixture; weights not loaded during pack/TypeScript tests']),
}))
export type PackedUniversalRetrievalInput = UniversalRetrievalInput

// Importing the public symbols cannot load model weights automatically.
export const faceSegmentationModelId: string = FACE_PARSING_MODEL.id
export const pointSegmentationVariants = SAM2_MODEL_VARIANTS
export const externalFaceSegmentation = suggestFaceRegions
export const externalPointSegmentation = selectSamPoint
export const externalSignedPointSegmentation = selectSamPoints
export const externalPrepareSamPoints = prepareSamPointPrompts
export const externalSignedPoint: SamPointPrompt = { x: 5, y: 7, label: 0 }
export const foregroundPipeline = new SaliencySegmentationPipeline()
export const foregroundManifest = u2NetSaliencyManifest
export const foregroundLayerFactory = extractMaskLayer
export type ExternalForegroundMask = ForegroundMask
export const externalValidateRegion = validateRegion
export type ExternalPrivacyResult = PrivacyResult
export type ExternalSegmentationMask = SegmentationRegion
export type ExternalSam2Variant = Sam2Variant

// The moderation adapter stays lazy even when imported from a packed consumer:
// nothing downloads until the consumer explicitly loads its runner.
export const externalModerationSubstrate = createModerationSubstrate()
export const externalModerationRunner = new BrowserSubstrateRunner(externalModerationSubstrate)
export const externalModerationModelId: string = MODERATION_MODEL_ID
export const externalModerationRevision: string = MODERATION_MODEL_REVISION
export const externalModerationCategories = MODERATION_CATEGORIES
export type ExternalModerationResult = ModerationResult

// Substrate-neutral public API must work from a *packed* consumer, without
// pulling in a specific LiteRT/ONNX/GGUF implementation at import time.
export const externalSubstrate: BrowserSubstrate<string, string> = {
  id: 'external-test-provider',
  format: 'test',
  capabilities: ['text-generation'],
  requestedBackends: ['wasm'],
  async create() {
    return { infer: async (input) => input, dispose: async () => undefined }
  },
}
export const externalSubstrateRunner = new BrowserSubstrateRunner(externalSubstrate)

// The embedding adapter stays lazy even when imported from a packed consumer.
export const externalEmbeddingPrompt = formatEmbeddingGemma2Text('remote document', 'search-query')
export const externalEmbeddingSubstrate = createEmbeddingGemma2Substrate({
  model: new Blob(['caller-owned fixture; no model load until explicit load()']),
})
export const externalEmbeddingRunner = new BrowserSubstrateRunner(externalEmbeddingSubstrate)

// A constructor option whose type is unreachable from the entrypoint is a hole in the
// supported surface, so the packed consumer has to be able to name it.
export const textPipelineOptions: LiteRtLmTextPipelineOptions = {
  modelBase: 'https://huggingface.co/',
}

// Paper-inspired speculation policy is a public, runtime-neutral text-generation
// primitive. Importing it must not initialize a model, GPU runtime, or browser store.
export const externalDraftCandidates = collectSuffixDraftCandidates(
  [1, 2, 3],
  [[9, 1, 2, 3, 4, 5]],
)
export const externalSessionMemory = new BoundedSessionTokenMemory(64)
export const externalSpeculationController = new AdaptiveSpeculationController([1, 2, 4])
export const externalSubmissionWindow = chooseSubmissionWindow([
  { steps: 1, elapsedMs: 10 },
  { steps: 2, elapsedMs: 18 },
])

// The lightweight downstream loader is part of the supported public surface:
// consumers with their own model cache only need to provide the execution backend.
export async function loadCachedTextModel(model: Blob): Promise<void> {
  const pipeline = new LiteRtLmTextPipeline('gemma-4-e2b-it', { model })
  await pipeline.loadForBackend('webgpu')
  await pipeline.dispose()
}

export const multimodalWorkerLoadOptions: LiteRtLmWorkerLoadOptions = {
  engine: 'mediapipe',
  mediaPipe: { maxNumImages: 1, supportAudio: true },
}

export const depthFrameContract: Pick<
  DepthFrame,
  'kind' | 'width' | 'height' | 'sourceToDepthUv' | 'nearFarConvention'
> = {
  kind: 'depth',
  width: 1,
  height: 1,
  sourceToDepthUv: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
  nearFarConvention: 'unknown',
}

export const exampleD1Question: D1Question = {
  type: 'noul',
  instructions: 'A source-stamped observation exists.',
}
export const d1TextRuntime = D1OnnxWebGpu

// Importing the packed Pix2Pix package must remain lazy: the Wasm module and
// caller-owned 217 MB weights are untouched until an explicit runner load.
export const externalPix2PixManifest = pix2pixEdge2ShoesManifest
export const externalPix2PixSubstrate = createPix2PixEdge2ShoesSubstrate({
  assets: { async resolve() { return new ArrayBuffer(0) } },
})
export const externalPix2PixRunner = new BrowserSubstrateRunner(externalPix2PixSubstrate)

export const depthPipeline = new DepthEstimationPipeline()
export const depthManifest = da3SmallPortraitManifest

export const multimodalWorkerPrompt: LiteRtLmWorkerPrompt = [
  { type: 'text', text: 'Describe this frame.' },
  { type: 'image', data: new Blob(['frame']) },
]

export const importedPackages = [
  InferenceError,
  createHttpAssetResolver,
  createLiteRtRuntime,
  ManagedLiteRtModelController,
  useManagedLiteRtModel,
  createBrowserCacheAssetResolver,
  LiteRtLmTextPipeline,
  LiteRtLmWorkerClient,
  AdaptiveSpeculationController,
  BoundedSessionTokenMemory,
  chooseSubmissionWindow,
  collectSuffixDraftCandidates,
  EncoderPipeline,
  createPrivacyFilterSubstrate,
  redactPrivacySpans,
  D1OnnxWebGpu,
  createPix2PixEdge2ShoesSubstrate,
  ColBertPipeline,
  maxSim,
  rankDense,
  KokoroPipeline,
  PhononPipeline,
  Qwen3TtsPipeline,
  ClipImageEmbeddingPipeline,
  MoViNetPipeline,
  DepthEstimationPipeline,
  da3SmallPortraitManifest,
  createModerationSubstrate,
]

// MoGe is exposed through the packed public entrypoint; consumers own runtime,
// selection and frame provenance rather than reaching into internal sources.
export const moGeEstimate = estimateMoGeGeometry
export const moGeDispose = releaseMoGeModels
export const moGeCanonicalDepth = moGeGeometryToDepthFrame
export const moGePinnedRevision: string = MOGE_REVISION
export type ExternalMoGeGeometry = MoGeGeometry
