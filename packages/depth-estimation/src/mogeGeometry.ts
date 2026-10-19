import type { DepthFrame } from '@lil-local/inference-core';

/**
 * MoGe-2 ViT-S tensor contract. No inferred depth values or simulated geometry.
 * Source: litert-community/MoGe-2-LiteRT (MIT), revision pinned by mogeGeometryService.
 */
export const MOGE_SIZE = 448;
export const MOGE_PIXELS = MOGE_SIZE * MOGE_SIZE;
export const MOGE_MODEL_ID = 'litert-community/MoGe-2-LiteRT';

export interface MoGeGeometry {
  modelId: string;
  modelRevision: string;
  backend: 'webgpu' | 'wasm';
  inputWidth: typeof MOGE_SIZE;
  inputHeight: typeof MOGE_SIZE;
  sourceWidth: number;
  sourceHeight: number;
  frameId?: string | number;
  inferenceMs: number;
  /** Affine camera-space points; NOT calibrated metric XYZ until focal/shift recovery. */
  affinePoints: Float32Array;
  /** L2-normalized surface normals in the model's coordinate system. */
  normals: Float32Array;
  /** Original sigmoid values; >0.5 is the model's valid-geometry threshold. */
  mask: Float32Array;
  /** Relative min/max normalized Z of valid points. Larger = farther; NOT meters. */
  relativeDepth: Float32Array;
  depthMin: number;
  depthMax: number;
  validFraction: number;
  /** Raw model scale prediction, NOT an independently measured depth calibration. */
  scaleFactor: number;
  /** Original source UV maps to same UV in square-resized model input. */
  sourceToModelUV: {scaleX: 1; scaleY: 1; offsetX: 0; offsetY: 0};
}

export type MoGeTensorData = {
  shape: readonly number[];
  values: Float32Array;
  name?: string;
};

function requireShape(t: MoGeTensorData, shape: readonly number[]): void {
  if (t.shape.length !== shape.length || !shape.every((n, i) => n === t.shape[i])) {
    throw new Error('MoGe-2 tensor shape mismatch: ' + t.shape.join('x'));
  }
  if (t.values.length !== shape.reduce((a, n) => a * n, 1)) {
    throw new Error('MoGe-2 tensor element count mismatch');
  }
}

function normalError(values: Float32Array): number {
  // Sparse sampling is sufficient to identify the two equal-shaped heads:
  // normal vectors have unit L2 norm, affine points do not.
  const count = MOGE_PIXELS;
  let error = 0;
  let sampled = 0;
  for (let p = 0; p < count; p += 613) {
    const i = p * 3;
    const norm = Math.hypot(values[i], values[i + 1], values[i + 2]);
    if (!Number.isFinite(norm)) throw new Error('MoGe-2 returned nonfinite geometry');
    error += Math.abs(norm - 1);
    sampled += 1;
  }
  return error / sampled;
}

export function decodeMoGeOutputs(
  outputs: readonly MoGeTensorData[],
  meta: Pick<MoGeGeometry, 'backend' | 'modelId' | 'modelRevision' | 'sourceWidth' | 'sourceHeight' | 'inferenceMs' | 'frameId'>,
): MoGeGeometry {
  if (outputs.length !== 4) throw new Error('MoGe-2 must produce exactly four outputs');
  const vectors = outputs.filter((o) => o.shape.join(',') === [1, MOGE_SIZE, MOGE_SIZE, 3].join(','));
  const masks = outputs.filter((o) => o.shape.join(',') === [1, MOGE_SIZE, MOGE_SIZE, 1].join(','));
  const scales = outputs.filter((o) => o.shape.join(',') === '1,1,1,1');
  if (vectors.length !== 2 || masks.length !== 1 || scales.length !== 1) {
    throw new Error('MoGe-2 outputs do not match the published four-head contract');
  }
  for (const o of vectors) requireShape(o, [1, MOGE_SIZE, MOGE_SIZE, 3]);
  requireShape(masks[0], [1, MOGE_SIZE, MOGE_SIZE, 1]);
  requireShape(scales[0], [1, 1, 1, 1]);
  const err0 = normalError(vectors[0].values);
  const err1 = normalError(vectors[1].values);
  if (Math.min(err0, err1) > 0.08 || Math.abs(err0 - err1) < 0.025) {
    throw new Error('Cannot reliably distinguish MoGe-2 normals from affine points');
  }
  const normals = err0 < err1 ? vectors[0].values : vectors[1].values;
  const affinePoints = err0 < err1 ? vectors[1].values : vectors[0].values;
  const mask = masks[0].values;
  const scaleFactor = scales[0].values[0];
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
    throw new Error('MoGe-2 returned an invalid scale factor');
  }
  const relativeDepth = new Float32Array(MOGE_PIXELS);
  let min = Infinity;
  let max = -Infinity;
  let valid = 0;
  for (let i = 0; i < MOGE_PIXELS; i += 1) {
    if (!Number.isFinite(mask[i]) || mask[i] < 0 || mask[i] > 1) {
      throw new Error('MoGe-2 returned an invalid confidence mask');
    }
    if (mask[i] <= 0.5) continue;
    const offset = i * 3;
    const z = affinePoints[offset + 2];
    for (let channel = 0; channel < 3; channel += 1) {
      if (!Number.isFinite(affinePoints[offset + channel]) ||
          !Number.isFinite(normals[offset + channel])) {
        throw new Error('MoGe-2 returned nonfinite geometry');
      }
    }
    min = Math.min(min, z);
    max = Math.max(max, z);
    valid += 1;
  }
  if (valid === 0) throw new Error('MoGe-2 did not produce valid geometry');
  const inv = max > min ? 1 / (max - min) : 0;
  for (let i = 0; i < MOGE_PIXELS; i += 1) {
    if (mask[i] > 0.5) relativeDepth[i] = (affinePoints[i * 3 + 2] - min) * inv;
  }
  return {
    ...meta,
    inputWidth: MOGE_SIZE,
    inputHeight: MOGE_SIZE,
    affinePoints,
    normals,
    mask,
    relativeDepth,
    depthMin: min,
    depthMax: max,
    scaleFactor,
    validFraction: valid / MOGE_PIXELS,
    sourceToModelUV: {scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0},
  };
}

/** Bilinear RGBA -> RGB float32 NCHW, [0,1]; square stretch matches model card. */
export function prepareMoGeImage(image: ImageData): Float32Array {
  const {width, height, data} = image;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
      data.length !== width * height * 4) {
    throw new Error('MoGe-2 requires valid nonempty RGBA ImageData');
  }
  const out = new Float32Array(MOGE_PIXELS * 3);
  for (let y = 0; y < MOGE_SIZE; y += 1) {
    const srcY = Math.max(0, Math.min(height - 1, (y + 0.5) * height / MOGE_SIZE - 0.5));
    const y0 = Math.floor(srcY);
    const y1 = Math.min(y0 + 1, height - 1);
    const fy = srcY - y0;
    for (let x = 0; x < MOGE_SIZE; x += 1) {
      const srcX = Math.max(0, Math.min(width - 1, (x + 0.5) * width / MOGE_SIZE - 0.5));
      const x0 = Math.floor(srcX);
      const x1 = Math.min(x0 + 1, width - 1);
      const fx = srcX - x0;
      const a = (y0 * width + x0) * 4;
      const b = (y0 * width + x1) * 4;
      const c = (y1 * width + x0) * 4;
      const d = (y1 * width + x1) * 4;
      const p = y * MOGE_SIZE + x;
      for (let ch = 0; ch < 3; ch += 1) {
        const top = data[a + ch] * (1 - fx) + data[b + ch] * fx;
        const bottom = data[c + ch] * (1 - fx) + data[d + ch] * fx;
        out[ch * MOGE_PIXELS + p] = (top * (1 - fy) + bottom * fy) / 255;
      }
    }
  }
  return out;
}

/** Canonical relative-depth view of the richer MoGe geometry result.
 * Invalid pixels are zeroed and their reliability remains in geometry.mask.
 * timestampMs is caller-provided capture time if known, otherwise conversion
 * time — NOT a fabricated capture timestamp. */
export function moGeGeometryToDepthFrame(
  geometry: MoGeGeometry,
  timestampMs: number = Date.now(),
): DepthFrame {
  return {
    kind: 'depth',
    data: geometry.relativeDepth.slice(),
    width: geometry.inputWidth,
    height: geometry.inputHeight,
    sourceWidth: geometry.sourceWidth,
    sourceHeight: geometry.sourceHeight,
    sourceToDepthUv: {
      scaleX: geometry.sourceToModelUV.scaleX,
      scaleY: geometry.sourceToModelUV.scaleY,
      offsetX: geometry.sourceToModelUV.offsetX,
      offsetY: geometry.sourceToModelUV.offsetY,
    },
    ...(geometry.frameId === undefined ? {} : { frameId: geometry.frameId }),
    timestampMs,
    nearFarConvention: 'higher-far',
    normalization: 'frame-minmax',
    rawRange: {min: geometry.depthMin, max: geometry.depthMax},
    modelId: geometry.modelId,
    backend: geometry.backend,
    inferenceMs: geometry.inferenceMs,
  };
}
