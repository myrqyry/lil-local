export { DepthEstimationPipeline } from './pipeline';
export {
  da3SmallPortraitManifest,
  da3SmallPortraitSpec,
} from './manifest';
export {
  assertSourceAspectCompatible,
  createDepthFrame,
  inspectDepthTensorContract,
  normalizeDepthValues,
} from './contract';
export type { DepthConfig, DepthInput } from './pipeline';
export type {
  DepthFrameMetadata,
  DepthModelSpec,
  DepthTensorContract,
} from './contract';
export type {
  DepthFrame,
  DepthNearFarConvention,
  DepthUvTransform,
} from '@lil-local/inference-core';

export {
  estimateMoGeGeometry,
  releaseMoGeModels,
  MOGE_REVISION,
  MOGE_GPU_URL,
  MOGE_CPU_URL,
  type MoGeEstimateOptions,
} from './mogeGeometryService';
export { MOGE_SIZE, MOGE_PIXELS, MOGE_MODEL_ID, prepareMoGeImage, decodeMoGeOutputs, moGeGeometryToDepthFrame } from './mogeGeometry';
export type { MoGeGeometry, MoGeTensorData } from './mogeGeometry';
