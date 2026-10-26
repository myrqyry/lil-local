import type * as Ort from 'onnxruntime-web'
import { decodeSharpGaussians, type SharpGaussians, type SharpOutputTensors } from './geometry'
import { buildSharpPly } from './ply'

export const SHARP_RESOLUTION=1536
export interface SharpOnnxOptions {
  /** Caller-hosted model: never downloaded by import/discovery. */
  graphUrl: string
  externalDataUrl: string
  externalDataPath?: string
  precision?: 'fp32' | 'fp16'
}
export interface SharpRunInput {
  /** Planar RGB float32, shape [3,1536,1536], values 0..1. */
  rgb: Float32Array
  width: number
  height: number
  focalPx: number
  opacityThreshold?: number
  maxGaussians?: number
}
export interface SharpRunOutput {
  gaussians: SharpGaussians
  ply: Uint8Array
  modelUrl: string
  /** Requested EP; not measured per-node execution evidence. */
  requestedBackend: 'webgpu'
}
const outputs=['mean_vectors_ndc','singular_values_ndc','quaternions_ndc','colors','opacities']
function checkAbort(signal?:AbortSignal):void {
  if(!signal?.aborted)return
  const e=new Error('SHARP ONNX load aborted');e.name='AbortError';throw e
}
function halfBits(src:Float32Array):Uint16Array {
  const dst=new Uint16Array(src.length),buf=new ArrayBuffer(4),view=new DataView(buf)
  for(let j=0;j<src.length;j++){
    view.setFloat32(0,src[j])
    const u=view.getUint32(0),sign=(u>>>16)&0x8000
    let exp=((u>>>23)&255)-112, m=u&0x7fffff
    if(exp>=31){dst[j]=sign|0x7c00;continue}
    if(exp<=0){
      if(exp< -10){dst[j]=sign;continue}
      m=(m|0x800000)>>(1-exp)
      dst[j]=sign|((m+0x1000)>>13);continue
    }
    m+=0x1000
    if(m&0x800000){m=0;exp++}
    dst[j]=exp>=31?sign|0x7c00:sign|(exp<<10)|(m>>13)
  }
  return dst
}
/** Caller can use this helper in an ordinary browser, or prepare RGB inside a worker. */
export function sharpImageToRgb(image:ImageBitmap):Float32Array {
  if(typeof document==='undefined')throw new Error('sharpImageToRgb requires a Canvas; provide rgb directly in workers')
  const canvas=document.createElement('canvas')
  canvas.width=canvas.height=SHARP_RESOLUTION
  const ctx=canvas.getContext('2d',{willReadFrequently:true})
  if(!ctx)throw new Error('Cannot get Canvas2D context')
  ctx.drawImage(image,0,0,SHARP_RESOLUTION,SHARP_RESOLUTION)
  const pixels=ctx.getImageData(0,0,SHARP_RESOLUTION,SHARP_RESOLUTION).data
  const n=SHARP_RESOLUTION*SHARP_RESOLUTION, rgb=new Float32Array(n*3)
  for(let j=0;j<n;j++){rgb[j]=pixels[j*4]/255;rgb[n+j]=pixels[j*4+1]/255;rgb[2*n+j]=pixels[j*4+2]/255}
  return rgb
}

/** No implicit CPU fallback. Output is raw Gaussian data and a ready-to-render PLY. */
export class SharpOnnxWebGpu {
  private session:Ort.InferenceSession|null=null
  private ort:typeof Ort|null=null
  private disposed=false
  private busy=false
  private loading=false
  private disposing:Promise<void>|null=null
  private idle:Array<()=>void>=[]
  constructor(readonly options:SharpOnnxOptions) {
    if(!options.graphUrl || !options.externalDataUrl)throw new Error('SHARP needs explicit graph and weight URLs')
  }
  get ready():boolean{return !!this.session && !this.disposed}
  async load(signal?:AbortSignal):Promise<void> {
    if(this.loading)throw new Error('SHARP load already in progress')
    this.loading=true
    try { await this.loadInternal(signal) }
    finally { this.loading=false }
  }
  private async loadInternal(signal?:AbortSignal):Promise<void> {
    if(this.disposed)throw new Error('SHARP runner disposed')
    if(this.session)throw new Error('SHARP model already loaded')
    checkAbort(signal)
    if(typeof navigator==='undefined' || !navigator.gpu)throw new Error('SHARP requires WebGPU; no silent WASM fallback')
    const ort=await import('onnxruntime-web/webgpu')
    checkAbort(signal)
    const pathname=new URL(this.options.graphUrl,globalThis.location?.href??'http://localhost/').pathname
    const sidecar=this.options.externalDataPath ?? pathname.substring(pathname.lastIndexOf('/')+1)+'.data'
    let session:Ort.InferenceSession|null=null
    try {
      session=await ort.InferenceSession.create(this.options.graphUrl,{
        executionProviders:['webgpu'],graphOptimizationLevel:'all',
        externalData:[{path:sidecar,data:this.options.externalDataUrl}],
      })
      checkAbort(signal)
      if(this.disposed)throw new Error('SHARP runner disposed during load')
      if(session.inputNames.length!==2 ||
        !session.inputNames.includes('image') || !session.inputNames.includes('disparity_factor') ||
        outputs.some(k=>!session!.outputNames.includes(k))){
        throw new Error('SHARP ONNX graph does not match raw predictor ABI')
      }
      this.session=session;this.ort=ort
    }catch(error){
      if(session){
        try{await session.release()}
        catch(cleanup){throw Object.assign(new Error('SHARP load and cleanup failed'),{errors:[error,cleanup]})}
      }
      throw error
    }
  }
  async run(input:SharpRunInput):Promise<SharpRunOutput> {
    if(!this.ready || !this.session || !this.ort)throw new Error('SHARP model is not loaded')
    if(this.busy)throw new Error('SHARP inference already running')
    const {rgb,width,height,focalPx}=input
    if(rgb.length!==3*SHARP_RESOLUTION*SHARP_RESOLUTION)throw new Error('SHARP expects [3,1536,1536] planar RGB')
    if(![width,height,focalPx].every(n=>Number.isFinite(n)&&n>0))throw new Error('Invalid SHARP camera')
    this.busy=true
    try {
      const ort=this.ort,session=this.session
      const fp16=this.options.precision==='fp16'
      const tensor=(v:Float32Array,dims:number[])=>fp16
        ? new ort.Tensor('float16',halfBits(v),dims):new ort.Tensor('float32',v,dims)
      const raw=await session.run({
        image:tensor(rgb,[1,3,SHARP_RESOLUTION,SHARP_RESOLUTION]),
        disparity_factor:tensor(new Float32Array([focalPx/width]),[1]),
      })
      const converted=Object.fromEntries(outputs.map(k=>{
        const t=raw[k]
        if(!t)throw new Error(`SHARP missing output ${k}`)
        return [k,{type:t.type,dims:t.dims,data:t.data}]
      })) as unknown as SharpOutputTensors
      const gaussians=decodeSharpGaussians(converted,{
        width,height,focalPx,opacityThreshold:input.opacityThreshold,maxGaussians:input.maxGaussians,
      })
      return {gaussians,ply:buildSharpPly(gaussians),modelUrl:this.options.graphUrl,requestedBackend:'webgpu'}
    }finally{
      this.busy=false;for(const done of this.idle.splice(0))done()
    }
  }
  dispose():Promise<void> {
    if(this.disposing)return this.disposing
    this.disposed=true
    this.disposing=(async()=>{
      if(this.busy)await new Promise<void>(resolve=>this.idle.push(resolve))
      const session=this.session;this.session=null;this.ort=null
      if(session)await session.release()
    })()
    return this.disposing
  }
}
