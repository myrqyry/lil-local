/** Original-image neutral, labeled single-channel mask output; no product/rig assumptions. */
export interface SegmentationRegion {
  label: string
  width: number
  height: number
  /** Row-major alpha: 0 transparent, 255 opaque. */
  data: Uint8Array
  /** Optional model-estimated IoU quality (not calibrated ground-truth accuracy). */
  score?: number
}
export type Point = Readonly<{ x: number; y: number }>
export type ProgressCallback = (message: string) => void
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024
export const MAX_SOURCE_PIXELS = 4 * 1024 * 1024

const FACE_LABELS = new Set([
  'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
  'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',
  'ear_r', 'neck_l', 'neck', 'cloth',
])

export function normalizeFaceMaskLabel(label: string | null): string | null {
  const name = (label ?? '').replace(/\.png$/i, '')
  return FACE_LABELS.has(name) ? name : null
}

export function validateSegmentationSource(file: Blob): void {
  if (!(file instanceof Blob) || !['image/png','image/jpeg','image/webp'].includes(file.type)) {
    throw new Error('Choose a supported PNG, JPEG or WebP illustration')
  }
  if (file.size > MAX_IMAGE_BYTES) throw new Error('Illustration exceeds 20 MiB')
}
export function validateSourceDimensions(width: number, height: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 ||
      height < 1 || width > 8192 || height > 8192 ||
      width * height > MAX_SOURCE_PIXELS) {
    throw new Error('Image must contain at most four million pixels')
  }
}
export function validatePointInImage(point: Point, width: number, height: number): void {
  validateSourceDimensions(width, height)
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) ||
      point.x < 0 || point.y < 0 || point.x >= width || point.y >= height) {
    throw new Error('Point lies outside the source image')
  }
}
export function validateRegion(region: SegmentationRegion): void {
  const { width, height, data } = region
  if (!Number.isInteger(width) || !Number.isInteger(height) ||
      width < 1 || height < 1 || width > 1024 || height > 1024 ||
      width * height > 1024 * 1024 ||
      !(data instanceof Uint8Array) || data.length !== width * height) {
    throw new Error('Invalid segmentation mask dimensions or data')
  }
}
