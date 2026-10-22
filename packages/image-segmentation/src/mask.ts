/** Browser-independent foreground segmentation data, deliberately separate from rig/semantic part labels. */
export interface RgbaSource {
  width: number
  height: number
  data: Uint8ClampedArray
}
export interface ForegroundMask {
  kind: 'foreground-mask'
  width: number
  height: number
  sourceWidth: number
  sourceHeight: number
  alpha: Uint8ClampedArray
  modelId?: string
  backend?: 'wasm' | 'webgpu' | 'webnn'
}
export interface MaskedLayer {
  x: number
  y: number
  width: number
  height: number
  data: Uint8ClampedArray
}
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]
const MAX_SOURCE_PIXELS = 4_194_304

function validateDimensions(width: number, height: number, maxPixels = MAX_SOURCE_PIXELS): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
    width < 1 || height < 1 || width > 8192 || height > 8192 ||
    width * height > maxPixels) {
    throw new RangeError('Image dimensions exceed bounded source pixels')
  }
}
function validateSource(source: RgbaSource): void {
  validateDimensions(source.width, source.height)
  if (!(source.data instanceof Uint8ClampedArray) ||
    source.data.length !== source.width * source.height * 4) {
    throw new TypeError('Expected complete RGBA pixels')
  }
}
function sampleChannel(source: Uint8ClampedArray, width: number, height: number,
  x: number, y: number, channel: number, stride: number): number {
  const x0 = Math.min(width - 1, Math.max(0, Math.floor(x)))
  const y0 = Math.min(height - 1, Math.max(0, Math.floor(y)))
  const x1 = Math.min(width - 1, x0 + 1)
  const y1 = Math.min(height - 1, y0 + 1)
  const wx = Math.max(0, Math.min(1, x - x0))
  const wy = Math.max(0, Math.min(1, y - y0))
  const a = source[(y0 * width + x0) * stride + channel]
  const b = source[(y0 * width + x1) * stride + channel]
  const c = source[(y1 * width + x0) * stride + channel]
  const d = source[(y1 * width + x1) * stride + channel]
  return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
}

/**
 * Follow Lil Local's U²-Net saliency adapter: image-max RGB normalization,
 * ImageNet mean/std and 320×320 NCHW. Optional dimensions enable tiny tests.
 */
export function prepareSaliencyInput(source: RgbaSource, width = 320, height = 320): Float32Array {
  validateSource(source)
  validateDimensions(width, height)
  const pixels = width * height
  const rgb = new Float32Array(3 * pixels)
  let max = 0
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sourceX = (x + .5) * source.width / width - .5
    const sourceY = (y + .5) * source.height / height - .5
    const index = y * width + x
    for (let c = 0; c < 3; c++) {
      const value = sampleChannel(source.data, source.width, source.height, sourceX, sourceY, c, 4)
      rgb[c * pixels + index] = value
      if (value > max) max = value
    }
  }
  const scale = max || 1
  for (let c = 0; c < 3; c++) for (let i = 0; i < pixels; i++) {
    rgb[c * pixels + i] = (rgb[c * pixels + i] / scale - MEAN[c]) / STD[c]
  }
  return rgb
}

export function decodeSaliencyMask(
  tensor: Float32Array, width: number, height: number,
  sourceWidth: number, sourceHeight: number,
): ForegroundMask {
  validateDimensions(width, height)
  validateDimensions(sourceWidth, sourceHeight)
  if (!(tensor instanceof Float32Array) || tensor.length !== width * height) {
    throw new TypeError('Unexpected saliency mask length')
  }
  const alpha = new Uint8ClampedArray(tensor.length)
  for (let i = 0; i < tensor.length; i++) {
    if (!Number.isFinite(tensor[i])) throw new Error('Saliency mask values must be finite')
    alpha[i] = Math.round(Math.min(1, Math.max(0, tensor[i])) * 255)
  }
  return {kind:'foreground-mask',width,height,sourceWidth,sourceHeight,alpha}
}

/** Apply the soft foreground matte to original RGB, retain original alpha and positional crop. */
export function extractMaskLayer(source: RgbaSource, mask: ForegroundMask): MaskedLayer {
  validateSource(source)
  if (source.width !== mask.sourceWidth || source.height !== mask.sourceHeight) {
    throw new Error('Mask source dimensions do not match the selected source image')
  }
  validateDimensions(mask.width, mask.height)
  if (mask.alpha.length !== mask.width * mask.height) {
    throw new TypeError('Mask alpha length is invalid')
  }
  // Source-sized working alpha only; bounded to 4 million pixels by validateSource.
  const sourceAlpha = new Uint8ClampedArray(source.width * source.height)
  let left = source.width
  let top = source.height
  let right = -1
  let bottom = -1
  for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) {
    const maskX = (x + .5) * mask.width / source.width - .5
    const maskY = (y + .5) * mask.height / source.height - .5
    const opacity = sampleChannel(mask.alpha, mask.width, mask.height, maskX, maskY, 0, 1)
    const index = y * source.width + x
    const output = Math.round(source.data[index * 4 + 3] * opacity / 255)
    sourceAlpha[index] = output
    if (output > 0) {
      left = Math.min(left, x)
      right = Math.max(right, x)
      top = Math.min(top, y)
      bottom = Math.max(bottom, y)
    }
  }
  if (right < 0) throw new Error('The foreground mask contains no visible pixels')
  const width = right - left + 1
  const height = bottom - top + 1
  const out = new Uint8ClampedArray(width * height * 4)
  for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
    const src = (y * source.width + x) * 4
    const dst = ((y - top) * width + x - left) * 4
    out.set(source.data.subarray(src, src + 3), dst)
    out[dst + 3] = sourceAlpha[y * source.width + x]
  }
  return {x:left,y:top,width,height,data:out}
}
