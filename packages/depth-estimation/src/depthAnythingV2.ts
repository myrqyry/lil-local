import type {BrowserSubstrate,DepthFrame} from '@lil-local/inference-core'
import {createDepthFrame} from './contract'

/** Exact upstream revision; fetching the model is explicit and inference stays local. */
export const DEPTH_ANYTHING_V2={
  id:'onnx-community/depth-anything-v2-small',
  revision:'4472b7362082ad9968fee890ca0f1e5aca36b93d',
  wasmDtype:'q8',
  webgpuDtype:'fp16',
} as const

export type DepthAnythingV2Image=ImageData|ImageBitmap|HTMLCanvasElement|HTMLImageElement|HTMLVideoElement
export interface DepthAnythingV2Frame {
  image:DepthAnythingV2Image
  timestampMs?:number
}
export interface DepthAnythingV2Options {
  /** Consumer-owned, self-hosted matching ONNX Runtime folder. No CDN execution. */
  wasmBase:string
  /** Explicit request; no silent backend fallback. */
  backend?:'wasm'|'webgpu'
}
export type DepthAnythingV2Progress='wasm'|'model'

export function onnxRuntimeLocalPaths(origin:string,base:string){
  if(!base.trim())throw new Error('ONNX runtime base is required')
  const resolved=new URL(base.endsWith('/')?base:base+'/',origin)
  if(resolved.origin!==new URL(origin).origin)
    throw new Error('ONNX runtime assets must be hosted on the serving origin')
  return {
    mjs:new URL('ort-wasm-simd-threaded.asyncify.mjs',resolved).href,
    wasm:new URL('ort-wasm-simd-threaded.asyncify.wasm',resolved).href,
  }
}

function sourceSize(image:DepthAnythingV2Image):{width:number;height:number}{
  if(typeof HTMLVideoElement!=='undefined' && image instanceof HTMLVideoElement)
    return {width:image.videoWidth,height:image.videoHeight}
  if(typeof HTMLImageElement!=='undefined' && image instanceof HTMLImageElement)
    return {width:image.naturalWidth,height:image.naturalHeight}
  return {width:image.width,height:image.height}
}
function abortError():Error{
  const error=new Error('Depth model loading aborted')
  error.name='AbortError'
  return error
}

/** Shared browser substrate. Reuses the existing Lil Local DepthFrame contract. */
export function createDepthAnythingV2Substrate(
  options:DepthAnythingV2Options,
):BrowserSubstrate<DepthAnythingV2Frame,DepthFrame,DepthAnythingV2Progress>{
  if(!options?.wasmBase?.trim())throw new Error('Self-hosted ONNX runtime base is required')
  const backend=options.backend??'wasm'
  if(backend!=='wasm'&&backend!=='webgpu')throw new Error('Unsupported depth backend')
  return {
    id:'onnx:depth-anything-v2-small-'+backend,
    format:'onnx',
    capabilities:['image-depth'],
    requestedBackends:[backend],
    async create(progress,signal){
      if(signal?.aborted)throw abortError()
      progress?.('wasm')
      const {pipeline,RawImage,env}=await import('@huggingface/transformers')
      if(signal?.aborted)throw abortError()
      if(!env.backends.onnx.wasm)throw new Error('ONNX WASM backend not available')
      // Transformers.js 4 would otherwise import its ONNX factory from a CDN.
      // The optional WASM preloader generates blob: imports, which stricter CSPs block.
      env.useWasmCache=false
      env.backends.onnx.wasm.wasmPaths=onnxRuntimeLocalPaths(location.origin,options.wasmBase)
      progress?.('model')
      const estimator=await pipeline('depth-estimation',DEPTH_ANYTHING_V2.id,{
        revision:DEPTH_ANYTHING_V2.revision,
        device:backend,
        dtype:backend==='wasm'?DEPTH_ANYTHING_V2.wasmDtype:DEPTH_ANYTHING_V2.webgpuDtype,
      })
      if(signal?.aborted){
        await estimator.dispose()
        throw abortError()
      }
      return {
        async infer(input){
          const {width,height}=sourceSize(input.image)
          if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||
            width<1||height<1||width*height>4_194_304)
            throw new Error('Depth source dimensions must be positive and within 4M pixels')
          if(input.timestampMs!==undefined&&
            (!Number.isFinite(input.timestampMs)||input.timestampMs<0))
            throw new Error('Depth frame timestamp must be a finite nonnegative millisecond value')
          const canvas=typeof OffscreenCanvas!=='undefined'
            ?new OffscreenCanvas(width,height)
            :Object.assign(document.createElement('canvas'),{width,height})
          const context=canvas.getContext('2d',{willReadFrequently:true}) as
            OffscreenCanvasRenderingContext2D|CanvasRenderingContext2D|null
          if(!context)throw new Error('Depth preprocessing 2D context unavailable')
          if(typeof ImageData!=='undefined'&&input.image instanceof ImageData)
            context.putImageData(input.image,0,0)
          else context.drawImage(input.image as CanvasImageSource,0,0,width,height)
          const rgba=context.getImageData(0,0,width,height)
          const start=performance.now()
          const predicted=await estimator(new RawImage(new Uint8Array(rgba.data),width,height,4))
          const tensor=predicted.predicted_depth
          const dims=tensor.dims
          const h=dims[dims.length-2],w=dims[dims.length-1]
          if(!Number.isSafeInteger(w)||!Number.isSafeInteger(h)||
            w<1||h<1||w*h>4_194_304||!(tensor.data instanceof Float32Array))
            throw new Error('Depth model output is not a bounded Float32 tensor')
          return createDepthFrame(tensor.data,w,h,{
            sourceWidth:width,sourceHeight:height,
            timestampMs:input.timestampMs??performance.now(),
            modelId:DEPTH_ANYTHING_V2.id,
            backend,
            nearFarConvention:'unknown',
            inferenceMs:performance.now()-start,
          })
        },
        async dispose(){await estimator.dispose()},
      }
    },
  }
}
