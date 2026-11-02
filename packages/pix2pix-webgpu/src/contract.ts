import type { ImageInput, ModelManifest } from '@lil-local/inference-core'

export const PIX2PIX_EDGE2SHOES_MODEL_ID = 'pix2pix-edges2shoes-unet256'
export const PIX2PIX_EDGE2SHOES_WEIGHT_BYTES = 217_677_836
export const PIX2PIX_MAJOR_STATIC_GPU_BYTES = 253_233_196
export const PIX2PIX_EDGE2SHOES_SOURCE_SHA256 =
  'a32f4a31ffd6c7c10ca6a28c022d38207480ea3e449822604a43a04c30db56f4'
export const PIX2PIX_EDGE2SHOES_FOLDED_SHA256 =
  'eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34'

export const PIX2PIX_EDGE2SHOES_WEIGHT_ASSET = {
  id: 'pix2pix-edges2shoes-folded-weights',
  path: 'pix2pix/edges2shoes-batchnorm-eval-v1.bin',
  bytes: PIX2PIX_EDGE2SHOES_WEIGHT_BYTES,
  sha256: PIX2PIX_EDGE2SHOES_FOLDED_SHA256,
  mimeType: 'application/octet-stream',
  role: 'weights',
} as const

export const pix2pixEdge2ShoesManifest: ModelManifest = {
  modelId: PIX2PIX_EDGE2SHOES_MODEL_ID,
  name: 'Pix2Pix U-Net 256 — edges2shoes',
  version: 'batchnorm-eval-fold-v1',
  capabilities: ['image-edit'],
  backends: { webgpu: 'experimental' },
  memory: {
    downloadBytes: PIX2PIX_EDGE2SHOES_WEIGHT_BYTES,
    // Major static GPU allocations derived from the verified engine topology.
    // Excludes browser/driver overhead and caller-held weight bytes.
    residentBytes: PIX2PIX_MAJOR_STATIC_GPU_BYTES,
  },
  assets: [PIX2PIX_EDGE2SHOES_WEIGHT_ASSET],
  verification: {
    assets: 'pass',
    compile: 'pass',
    inference: 'pass',
    output: 'pass',
    qualification: 'limited',
    lastVerifiedAt: '2026-11-01',
    environment: 'Linux Mint / Chromium WebGPU / NVIDIA Turing RTX 2070 Super',
    environments: [{
      browser: 'Chromium',
      backend: 'webgpu',
      runtime: 'Rust wgpu 23.0.1 + wasm-bindgen 0.2.130',
      device: 'NVIDIA Turing RTX 2070 Super',
    }],
    expectedOutput: {
      preprocessing: ['RGBA8 256x256', 'RGB normalized from [0,255] to [-1,1]'],
      outputShape: [3, 256, 256],
      behavior: ['paired image translation', 'final tanh output in [-1,1]'],
    },
  },
}

export function validatePix2PixInput(input: ImageInput): ImageInput {
  if (input.width !== 256 || input.height !== 256) {
    throw new Error('Pix2Pix input must be exactly 256x256 pixels')
  }
  if (!(input.data instanceof Uint8Array) || input.data.byteLength !== 256 * 256 * 4) {
    throw new Error('Pix2Pix input must contain exactly 262,144 RGBA8 bytes')
  }
  return input
}

export function nchwToRgba(values: Float32Array, width = 256, height = 256): Uint8ClampedArray {
  const plane = width * height
  if (values.length !== plane * 3) {
    throw new Error(`Pix2Pix output must contain exactly ${plane * 3} NCHW float values`)
  }
  const rgba = new Uint8ClampedArray(plane * 4)
  for (let i = 0; i < plane; i++) {
    for (let channel = 0; channel < 3; channel++) {
      const value = Math.max(-1, Math.min(1, values[channel * plane + i]))
      rgba[i * 4 + channel] = Math.round((value * 0.5 + 0.5) * 255)
    }
    rgba[i * 4 + 3] = 255
  }
  return rgba
}
