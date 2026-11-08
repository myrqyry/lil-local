/** Session-only evidence for real Live Vision still-image inference.
 * Nothing is written to storage and no model is loaded by this module.
 */
export const MAX_EXPERIMENTS = 4
const MAX_SAMPLE_SIDE = 256
const MAX_INPUT_PIXELS = 4_194_304

export interface SampledField {
  kind: 'person-mask' | 'depth'
  width: number
  height: number
  sourceWidth: number
  sourceHeight: number
  /** An owned, spatially sampled copy. Not the full native output tensor. */
  values: Float32Array
  minimum: number
  maximum: number
  mean: number
}

export interface VisionExperiment {
  id: number
  task: string
  requestedBackend: string
  image: { name: string; width: number; height: number; preview: string }
  capturedAt: number
  loadMs: number
  inferenceMs: number
  outputKind: string
  summary: string
  raster: SampledField | null
}

export interface ExperimentCapture<Output extends { kind: string }> {
  id: number
  task: string
  requestedBackend: string
  image: VisionExperiment['image']
  capturedAt: number
  loadMs: number
  inferenceMs: number
  output: Output
}

const positiveDimension = (value: number) =>
  Number.isSafeInteger(value) && value > 0

function sampleField(input: {
  kind: 'person-mask' | 'depth'
  width: number
  height: number
  values: Float32Array
}, maxSide: number): SampledField {
  const { width, height, values, kind } = input
  if (!positiveDimension(width) || !positiveDimension(height) ||
      width * height > MAX_INPUT_PIXELS)
    throw new Error('Invalid vision field dimensions')
  if (!(values instanceof Float32Array) || values.length !== width * height)
    throw new Error('Vision field length must equal the pixel dimensions')
  if (!positiveDimension(maxSide) || maxSide > MAX_SAMPLE_SIDE)
    throw new Error('Invalid sampling bound')

  const scale = Math.min(1, maxSide / Math.max(width, height))
  const targetWidth = Math.max(1, Math.round(width * scale))
  const targetHeight = Math.max(1, Math.round(height * scale))
  const sampled = new Float32Array(targetWidth * targetHeight)
  let minimum = Infinity, maximum = -Infinity, sum = 0
  // Validate every source pixel, even those not retained in this small preview.
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i]))
      throw new Error('Vision field must contain finite values')
  }
  for (let y = 0; y < targetHeight; y++) {
    const fromY = Math.floor(y * height / targetHeight)
    for (let x = 0; x < targetWidth; x++) {
      const value = values[fromY * width + Math.floor(x * width / targetWidth)]
      sampled[y * targetWidth + x] = value
      minimum = Math.min(minimum, value)
      maximum = Math.max(maximum, value)
      sum += value
    }
  }
  return {
    kind, width: targetWidth, height: targetHeight,
    sourceWidth: width, sourceHeight: height,
    values: sampled, minimum, maximum, mean: sum / sampled.length,
  }
}

function outputSummary(output: { kind: string }): string {
  const readCount = (name: string): number => {
    const value = (output as unknown as Record<string, unknown>)[name]
    return Array.isArray(value) ? value.length : 0
  }
  switch (output.kind) {
    case 'person-mask': return 'Foreground confidence map'
    case 'depth': return 'Relative depth map'
    case 'semantic-segmentation': return 'Semantic category map'
    case 'face-landmarks': return readCount('faces') + ' faces'
    case 'gestures': return readCount('hands') + ' hands'
    case 'pose-landmarks': return readCount('poses') + ' poses'
    case 'object-detections': return readCount('detections') + ' object detections'
    case 'image-classifications': return readCount('heads') + ' classification heads'
    case 'holistic-landmarks': return 'Holistic landmarks'
    case 'face-detections': return readCount('detections') + ' face detections'
    default: return 'Result: ' + output.kind
  }
}

/** Only real inference callers should invoke this; it does not synthesize outputs. */
export function createVisionExperiment<Output extends { kind: string }>(
  capture: ExperimentCapture<Output>,
  maxSide = MAX_SAMPLE_SIDE,
): VisionExperiment {
  const { id, task, requestedBackend, image, capturedAt, loadMs, inferenceMs, output } = capture
  if (!Number.isSafeInteger(id) || id < 0 ||
      !Number.isFinite(capturedAt) || capturedAt < 0 ||
      !Number.isFinite(loadMs) || loadMs < 0 ||
      !Number.isFinite(inferenceMs) || inferenceMs < 0)
    throw new Error('Invalid experiment identifier or timing')
  if (!positiveDimension(image.width) || !positiveDimension(image.height))
    throw new Error('Invalid source image dimensions')

  let raster: SampledField | null = null
  if (output.kind === 'person-mask' || output.kind === 'depth') {
    const typed = output as unknown as {
      kind: 'person-mask' | 'depth'
      width: number
      height: number
      confidence?: Float32Array
      data?: Float32Array
    }
    raster = sampleField({
      kind: typed.kind, width: typed.width, height: typed.height,
      values: typed.kind === 'person-mask' ? typed.confidence! : typed.data!,
    }, maxSide)
  }

  return {
    id, task, requestedBackend,
    image: { ...image }, capturedAt, loadMs, inferenceMs,
    outputKind: output.kind, summary: outputSummary(output), raster,
  }
}

export function appendExperiment(
  previous: readonly VisionExperiment[], next: VisionExperiment,
): VisionExperiment[] {
  return [next, ...previous.filter(entry => entry.id !== next.id)].slice(0, MAX_EXPERIMENTS)
}

export function renderSampledPixels(
  raster: SampledField,
  options: { mode: 'raw' | 'cutoff'; threshold: number },
): Uint8ClampedArray {
  if (!Number.isFinite(options.threshold) || options.threshold < 0 || options.threshold > 1)
    throw new Error('Confidence cutoff must be between 0 and 1')
  const rgba = new Uint8ClampedArray(raster.values.length * 4)
  for (let i = 0; i < raster.values.length; i++) {
    const v = raster.values[i]
    const pixel = raster.kind === 'person-mask' && options.mode === 'cutoff'
      ? (v >= options.threshold ? 255 : 0)
      : Math.round(Math.max(0, Math.min(1, v)) * 255)
    const pos = i * 4
    rgba[pos] = rgba[pos + 1] = rgba[pos + 2] = pixel
    rgba[pos + 3] = 255
  }
  return rgba
}
