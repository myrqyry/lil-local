/**
 * Opt-in browser inference. No model is loaded on import; weights are fetched
 * by Transformers.js only when an application calls one of these methods.
 * Host applications own cancellation via worker termination, UI and cutouts.
 */
import { FACE_PARSING_MODEL, getSam2ModelVariant, type Sam2Variant } from './models'
import {
  normalizeFaceMaskLabel, validatePointInImage, validateRegion,
  validateSegmentationSource, validateSourceDimensions,
  type ProgressCallback, type Point, type SegmentationRegion,
} from './contracts'
import { decodeSamMaskCandidates } from './masks'

async function readBoundedImage(file: Blob) {
  validateSegmentationSource(file)
  const { RawImage } = await import('@huggingface/transformers')
  const image = await RawImage.fromBlob(file)
  const originalWidth = image.width
  const originalHeight = image.height
  validateSourceDimensions(originalWidth, originalHeight)
  const ratio = Math.min(1, 512 / Math.max(originalWidth, originalHeight))
  if (ratio < 1) {
    await image.resize(Math.max(1, Math.round(originalWidth * ratio)),
      Math.max(1, Math.round(originalHeight * ratio)))
  }
  return { image, originalWidth, originalHeight }
}

export async function suggestFaceRegions(
  file: Blob,
  onProgress?: ProgressCallback,
): Promise<SegmentationRegion[]> {
  validateSegmentationSource(file)
  onProgress?.('Loading face-region parser on CPU (first use downloads weights)…')
  const { env, pipeline } = await import('@huggingface/transformers')
  env.allowLocalModels = false
  env.useBrowserCache = true
  const { image } = await readBoundedImage(file)
  const parser = await pipeline('image-segmentation', FACE_PARSING_MODEL.id, {
    revision: FACE_PARSING_MODEL.revision,
    device: FACE_PARSING_MODEL.device,
    dtype: FACE_PARSING_MODEL.dtype,
  })
  onProgress?.('Finding semantic face-region masks locally…')
  // Explicit subtask:'semantic' produced an unbound postprocessor error in
  // Transformers.js 4.2; relying on model-aware inference works in Chromium.
  const raw = await parser(image)
  const regions: SegmentationRegion[] = []
  for (const item of raw) {
    const label = normalizeFaceMaskLabel(item.label)
    const mask = item.mask
    if (!label || mask.channels !== 1) continue
    const region = {
      label, width: mask.width, height: mask.height,
      data: new Uint8Array(mask.data),
    }
    try { validateRegion(region) } catch { continue }
    if (region.data.some(value => value > 0)) regions.push(region)
  }
  if (!regions.length) throw new Error('No usable face regions found; use manual cutting')
  return regions
}

export async function selectSamPoint(
  file: Blob,
  point: Point,
  variant: Sam2Variant,
  onProgress?: ProgressCallback,
): Promise<SegmentationRegion[]> {
  validateSegmentationSource(file)
  const checkpoint = getSam2ModelVariant(variant)
  onProgress?.('Loading ' + checkpoint.label + ' on CPU (first use downloads weights)…')
  const { env, AutoProcessor, Sam2Model } = await import('@huggingface/transformers')
  env.allowLocalModels = false
  env.useBrowserCache = true
  const { image, originalWidth, originalHeight } = await readBoundedImage(file)
  validatePointInImage(point, originalWidth, originalHeight)
  const scaledPoint = [
    Math.min(image.width - 1, point.x * image.width / originalWidth),
    Math.min(image.height - 1, point.y * image.height / originalHeight),
  ]
  const model = await Sam2Model.from_pretrained(checkpoint.repo, {
    device: 'wasm', dtype: checkpoint.dtype, revision: checkpoint.revision,
  })
  const processor = await AutoProcessor.from_pretrained(checkpoint.repo, {
    revision: checkpoint.revision,
  })
  // Runtime returns SAM2Processor; static AutoProcessor type is generic.
  if (!('post_process_masks' in processor) ||
      typeof processor.post_process_masks !== 'function') {
    throw new Error('Pinned SAM checkpoint returned an incompatible processor')
  }
  onProgress?.('Selecting object masks around the clicked point…')
  const inputs = await processor(image, { input_points: [[scaledPoint]] })
  const output = await model(inputs)
  const masks = await processor.post_process_masks(
    output.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes,
  )
  if (!Array.isArray(masks) || masks.length !== 1) {
    throw new Error('SAM did not return a single-image mask result')
  }
  return decodeSamMaskCandidates(
    new Uint8Array(masks[0].data),
    masks[0].dims,
    Float32Array.from(output.iou_scores.data, Number),
  )
}
