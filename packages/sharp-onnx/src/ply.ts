import type { SharpGaussians } from './geometry'

const SH0 = 1 / Math.sqrt(4 * Math.PI)
function srgb(x:number):number {
  x=Math.min(1,Math.max(0,x))
  return x<=0.0031308 ? x*12.92 : 1.055*x**(1/2.4)-0.055
}
function quantile(values: Float32Array, fraction: number): number {
  if(!values.length) return 0
  const sorted=Array.from(values).sort((a,b)=>a-b)
  return sorted[Math.round((sorted.length-1)*fraction)]
}
/** Binary SHARP/SuperSplat PLY with camera metadata; no renderer dependency. */
export function buildSharpPly(g:SharpGaussians):Uint8Array {
  const header=[
    'ply','format binary_little_endian 1.0',`element vertex ${g.count}`,
    'property float x','property float y','property float z',
    'property float f_dc_0','property float f_dc_1','property float f_dc_2',
    'property float opacity','property float scale_0','property float scale_1',
    'property float scale_2','property float rot_0','property float rot_1',
    'property float rot_2','property float rot_3',
    'element extrinsic 16','property float extrinsic',
    'element intrinsic 9','property float intrinsic',
    'element image_size 2','property uint image_size',
    'element frame 2','property int frame',
    'element disparity 2','property float disparity',
    'element color_space 1','property uchar color_space',
    'element version 3','property uchar version','end_header','',
  ].join('\n')
  const encoded=new TextEncoder().encode(header)
  const output=new Uint8Array(encoded.byteLength+g.count*56+64+36+8+8+8+1+3)
  output.set(encoded)
  const view=new DataView(output.buffer)
  let ptr=encoded.byteLength
  const f=(x:number)=>{view.setFloat32(ptr,x,true);ptr+=4}
  const u=(x:number)=>{view.setUint32(ptr,x,true);ptr+=4}
  const i=(x:number)=>{view.setInt32(ptr,x,true);ptr+=4}
  const disparities=new Float32Array(g.count)
  for(let k=0;k<g.count;k++){
    const t=k*3, q=k*4
    if(!(g.scales[t]>0 && g.scales[t+1]>0 && g.scales[t+2]>0)) throw new Error('SHARP PLY scale must be positive')
    disparities[k]=1/Math.max(g.means[t+2],1e-6)
    for(let j=0;j<3;j++) f(g.means[t+j])
    for(let j=0;j<3;j++) f((srgb(g.colors[t+j])-0.5)/SH0)
    const opacity=Math.max(1e-6,Math.min(1-1e-6,g.opacities[k]))
    f(Math.log(opacity/(1-opacity)))
    for(let j=0;j<3;j++) f(Math.log(g.scales[t+j]))
    for(let j=0;j<4;j++) f(g.quaternions[q+j])
  }
  const {width,height,focalPx}=g.camera
  for(const n of [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]) f(n)
  for(const n of [focalPx,0,width/2,0,focalPx,height/2,0,0,1]) f(n)
  u(width);u(height)
  i(1);i(g.count)
  f(quantile(disparities,0.1));f(quantile(disparities,0.9))
  for(const byte of [1,1,5,0]) output[ptr++]=byte
  if(ptr!==output.byteLength) throw new Error('SHARP PLY byte count mismatch')
  return output
}
