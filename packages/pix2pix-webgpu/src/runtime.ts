import {
  createManifestVerifyingAssetResolver,
  type AssetIntegrityVerificationOptions,
  type AssetResolver,
  type BrowserSubstrate,
  type ImageInput,
  type InferenceReceipt,
} from '@lil-local/inference-core'
import {
  PIX2PIX_EDGE2SHOES_MODEL_ID,
  PIX2PIX_EDGE2SHOES_WEIGHT_ASSET,
  nchwToRgba,
  pix2pixEdge2ShoesManifest,
  validatePix2PixInput,
} from './contract'
import {
  loadBundledPix2PixWasm,
  type Pix2PixWasmRuntimeModule,
} from './wasm'

export interface Pix2PixLoadProgress {
  phase: 'weights' | 'wasm' | 'engine'
  loadedBytes?: number
  totalBytes?: number
}

export interface Pix2PixImageResult {
  kind: 'image'
  width: 256
  height: 256
  pixels: Uint8ClampedArray
  receipt: InferenceReceipt
}

export interface Pix2PixEdge2ShoesOptions {
  assets: AssetResolver
  canvas?: HTMLCanvasElement
  assetVerification?: AssetIntegrityVerificationOptions
  useTexture2DArraySkip?: boolean
}

type ProgressFn = (progress: Pix2PixLoadProgress) => void

export interface Pix2PixRuntimeDependencies {
  resolveWeights(
    options: Pix2PixEdge2ShoesOptions,
    signal?: AbortSignal,
    onProgress?: ProgressFn,
  ): Promise<ArrayBuffer>
  loadWasm(): Promise<Pix2PixWasmRuntimeModule>
  now(): number
}

function aborted(): Error {
  const error = new Error('Pix2Pix WebGPU load aborted')
  error.name = 'AbortError'
  return error
}

function checkAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw aborted()
}

function requireWebGpu(): void {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is unavailable; Pix2Pix load refused (no fallback backend)')
  }
}

function resolveCanvas(canvas?: HTMLCanvasElement): HTMLCanvasElement {
  if (canvas) {
    canvas.width = 256
    canvas.height = 256
    return canvas
  }
  if (typeof document === 'undefined') {
    throw new Error('Pix2Pix WebGPU requires a browser HTMLCanvasElement')
  }
  const created = document.createElement('canvas')
  created.width = 256
  created.height = 256
  return created
}

const defaultDependencies: Pix2PixRuntimeDependencies = {
  async resolveWeights(options, signal, onProgress) {
    const resolver = createManifestVerifyingAssetResolver(
      pix2pixEdge2ShoesManifest,
      options.assets,
      options.assetVerification,
    )
    return resolver.resolve(PIX2PIX_EDGE2SHOES_WEIGHT_ASSET, {
      signal,
      onProgress(progress) {
        onProgress?.({
          phase: 'weights',
          loadedBytes: progress.loadedBytes,
          totalBytes: progress.totalBytes,
        })
      },
    })
  },
  loadWasm: loadBundledPix2PixWasm,
  now: () => performance.now(),
}

export function createPix2PixEdge2ShoesSubstrate(
  options: Pix2PixEdge2ShoesOptions,
): BrowserSubstrate<ImageInput, Pix2PixImageResult, Pix2PixLoadProgress> {
  return createPix2PixEdge2ShoesSubstrateWithDeps(options, defaultDependencies)
}

/** Internal test seam; intentionally not exported from the package entrypoint. */
export function createPix2PixEdge2ShoesSubstrateWithDeps(
  options: Pix2PixEdge2ShoesOptions,
  deps: Pix2PixRuntimeDependencies,
): BrowserSubstrate<ImageInput, Pix2PixImageResult, Pix2PixLoadProgress> {
  return {
    id: 'rust-wgpu:pix2pix-edges2shoes-unet256',
    format: 'pix2pix-unet256-f32',
    capabilities: ['image-edit'],
    requestedBackends: ['webgpu'],
    async create(onProgress, signal) {
      checkAborted(signal)
      requireWebGpu()
      const canvas = resolveCanvas(options.canvas)

      const loadStart = deps.now()
      onProgress?.({ phase: 'weights', loadedBytes: 0 })
      const buffer = await deps.resolveWeights(options, signal, onProgress)
      const loadMs = deps.now() - loadStart
      checkAborted(signal)

      onProgress?.({ phase: 'wasm' })
      const wasm = await deps.loadWasm()
      checkAborted(signal)

      let engine
      const compileStart = deps.now()
      try {
        engine = await wasm.Pix2PixEngine.create_from_uint8_array(
          canvas,
          new Uint8Array(buffer),
          wasm.DropoutConvention.Disabled,
        )
        if (signal?.aborted) {
          engine.free()
          throw aborted()
        }
        engine.set_normalization_convention(
          wasm.NormalizationConvention.BatchNormEvalFolded,
        )
        engine.set_raw_pytorch_skip_order(false)
        engine.set_use_texture2d_array_skip(options.useTexture2DArraySkip ?? true)
      } catch (error) {
        if (engine && !signal?.aborted) engine.free()
        throw error
      }
      const compileMs = deps.now() - compileStart
      onProgress?.({ phase: 'engine' })

      return {
        async infer(input) {
          validatePix2PixInput(input)
          const start = deps.now()
          engine.run_frame(input.data)
          const output = await engine.read_output_nchw()
          const pixels = nchwToRgba(output)
          const inferenceMs = deps.now() - start
          return {
            kind: 'image',
            width: 256,
            height: 256,
            pixels,
            receipt: {
              modelId: PIX2PIX_EDGE2SHOES_MODEL_ID,
              pipelineVersion: '0.1.0',
              backend: 'webgpu',
              timestamp: new Date().toISOString(),
              loadMs,
              compileMs,
              inferenceMs,
              inputSummary: '256x256 RGBA8',
              outputSummary: '256x256 RGBA8',
              warnings: [
                'Custom WebGPU runtime; qualification is limited to the pinned folded edges2shoes weights.',
              ],
            },
          }
        },
        async dispose() {
          engine.free()
        },
      }
    },
  }
}
