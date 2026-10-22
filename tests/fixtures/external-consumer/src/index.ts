import { BrowserSubstrateRunner, InferenceError, createHttpAssetResolver, type BrowserSubstrate } from '@lil-local/inference-core'
import { createLiteRtRuntime } from '@lil-local/runtime-litert'
import { SaliencySegmentationPipeline, u2NetSaliencyManifest, extractMaskLayer, type ForegroundMask } from '@lil-local/image-segmentation'
import { ManagedLiteRtModelController, useManagedLiteRtModel } from '@lil-local/react-litert'
import { createBrowserCacheAssetResolver } from '@lil-local/browser-cache'
import {
  LiteRtLmTextPipeline,
  LiteRtLmWorkerClient,
  type LiteRtLmTextPipelineOptions,
  type LiteRtLmWorkerLoadOptions,
  type LiteRtLmWorkerPrompt,
} from '@lil-local/text-gen'
import { createEmbeddingGemma2Substrate, formatEmbeddingGemma2Text } from '@lil-local/embedding-litertlm'
import { EncoderPipeline } from '@lil-local/encoder'
import { D1OnnxWebGpu, type D1Question } from '@lil-local/decision-onnx'
import { ColBertPipeline } from '@lil-local/retrieval'
import { maxSim, rankDense } from '@lil-local/retrieval/scoring'
import { KokoroPipeline } from '@lil-local/kokoro'
import { Qwen3TtsPipeline } from '@lil-local/qwen3-tts'
import { ClipImageEmbeddingPipeline } from '@lil-local/image-embedding'
import { MoViNetPipeline } from '@lil-local/video-classification'
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
  EncoderPipeline,
  D1OnnxWebGpu,
  ColBertPipeline,
  maxSim,
  rankDense,
  KokoroPipeline,
  Qwen3TtsPipeline,
  ClipImageEmbeddingPipeline,
  MoViNetPipeline,
  DepthEstimationPipeline,
  da3SmallPortraitManifest,
]

// MoGe is exposed through the packed public entrypoint; consumers own runtime,
// selection and frame provenance rather than reaching into internal sources.
export const moGeEstimate = estimateMoGeGeometry
export const moGeDispose = releaseMoGeModels
export const moGeCanonicalDepth = moGeGeometryToDepthFrame
export const moGePinnedRevision: string = MOGE_REVISION
export type ExternalMoGeGeometry = MoGeGeometry

// Packed public masks remain substrate-agnostic. Never require model download on import.
export const foregroundPipeline = new SaliencySegmentationPipeline()
export const foregroundManifest = u2NetSaliencyManifest
export const foregroundLayerFactory = extractMaskLayer
export type ExternalForegroundMask = ForegroundMask
