/**
 * SHARP Gaussian decoding for the raw Apple predictor ONNX export.
 * No model weights bundled. The Gaussian covariance must be transformed when
 * NDC coordinates are unprojected; transforming centers alone is incorrect.
 */
export interface SharpTensor {
  type: string
  dims: readonly number[]
  data: ArrayLike<number>
}
export interface SharpOutputTensors {
  mean_vectors_ndc: SharpTensor
  singular_values_ndc: SharpTensor
  quaternions_ndc: SharpTensor
  colors: SharpTensor
  opacities: SharpTensor
}
export interface SharpCamera {
  width: number
  height: number
  focalPx: number
  opacityThreshold?: number
  maxGaussians?: number
}
export interface SharpGaussians {
  count: number
  totalCount: number
  means: Float32Array
  scales: Float32Array
  quaternions: Float32Array // wxyz
  colors: Float32Array // linear RGB
  opacities: Float32Array
  camera: { width: number; height: number; focalPx: number }
}
function halfToNumber(x: number): number {
  const sign = x & 0x8000 ? -1 : 1
  const exp = (x >> 10) & 0x1f
  const mant = x & 0x3ff
  if (exp === 0) return sign * 2 ** -14 * (mant / 1024)
  if (exp === 31) return mant ? NaN : sign * Infinity
  return sign * 2 ** (exp - 15) * (1 + mant / 1024)
}
function tensorValues(t: SharpTensor | undefined, name: string, stride: number): Float32Array {
  if (!t) throw new Error(`SHARP is missing output ${name}`)
  const dims = t.dims
  if (dims.length !== 3 || dims[0] !== 1 || dims[2] !== stride) {
    throw new Error(`SHARP ${name} shape must be [1, N, ${stride}]`)
  }
  if (t.data.length !== dims[1] * stride) throw new Error(`SHARP ${name} tensor count mismatch`)
  if (t.type !== 'float32' && t.type !== 'float16') throw new Error(`SHARP ${name} requires float16/float32`)
  const halfBits = t.type === 'float16' && t.data instanceof Uint16Array
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < data.length; i++) {
    const v = halfBits ? halfToNumber(Number(t.data[i])) : Number(t.data[i])
    if (!Number.isFinite(v)) throw new Error(`SHARP ${name} contains non-finite data`)
    data[i] = v
  }
  return data
}
function rotationFromQuaternion(w: number, x: number, y: number, z: number): number[] {
  const n = Math.hypot(w, x, y, z)
  if (!(n > 0)) throw new Error('SHARP quaternion has zero length')
  w /= n; x /= n; y /= n; z /= n
  return [
    1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
    2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
    2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
  ]
}
function quaternionFromRotation(r: number[]): [number, number, number, number] {
  const trace = r[0] + r[4] + r[8]
  let w: number, x: number, y: number, z: number
  if (trace > 0) {
    const s = 2 * Math.sqrt(1 + trace)
    w = s / 4; x = (r[7] - r[5]) / s; y = (r[2] - r[6]) / s; z = (r[3] - r[1]) / s
  } else if (r[0] > r[4] && r[0] > r[8]) {
    const s = 2 * Math.sqrt(1 + r[0] - r[4] - r[8])
    w = (r[7] - r[5]) / s; x = s / 4; y = (r[1] + r[3]) / s; z = (r[2] + r[6]) / s
  } else if (r[4] > r[8]) {
    const s = 2 * Math.sqrt(1 + r[4] - r[0] - r[8])
    w = (r[2] - r[6]) / s; x = (r[1] + r[3]) / s; y = s / 4; z = (r[5] + r[7]) / s
  } else {
    const s = 2 * Math.sqrt(1 + r[8] - r[0] - r[4])
    w = (r[3] - r[1]) / s; x = (r[2] + r[6]) / s; y = (r[5] + r[7]) / s; z = s / 4
  }
  return [w, x, y, z]
}
/** Eigenvectors are stored by columns. */
function diagonalize(a: number[]): { eigenvalues: number[]; basis: number[] } {
  const v = [1,0,0,0,1,0,0,0,1]
  for (let sweep = 0; sweep < 12; sweep++) {
    const pairs: Array<[number,number]> = [[0,1],[0,2],[1,2]]
    if (Math.abs(a[1]) + Math.abs(a[2]) + Math.abs(a[5]) < 1e-13) break
    for (const [p,q] of pairs) {
      const apq = a[p*3+q]
      if (Math.abs(apq) < 1e-16) continue
      const tau = (a[q*3+q]-a[p*3+p]) / (2*apq)
      const tangent = (tau >= 0 ? 1 : -1) / (Math.abs(tau)+Math.sqrt(1+tau*tau))
      const c = 1 / Math.sqrt(1+tangent*tangent), s = c*tangent
      const app = a[p*3+p], aqq = a[q*3+q]
      for(let k=0;k<3;k++) if(k!==p && k!==q) {
        const akp=a[k*3+p], akq=a[k*3+q]
        a[k*3+p]=a[p*3+k]=c*akp-s*akq
        a[k*3+q]=a[q*3+k]=s*akp+c*akq
      }
      a[p*3+p]=c*c*app-2*c*s*apq+s*s*aqq
      a[q*3+q]=s*s*app+2*c*s*apq+c*c*aqq
      a[p*3+q]=a[q*3+p]=0
      for(let k=0;k<3;k++) {
        const vp=v[k*3+p], vq=v[k*3+q]
        v[k*3+p]=c*vp-s*vq
        v[k*3+q]=s*vp+c*vq
      }
    }
  }
  const order=[0,1,2].sort((i,j)=>a[j*3+j]-a[i*3+i])
  const basis=Array.from({length:9},(_,i)=>v[Math.floor(i/3)*3+order[i%3]])
  const determinant=basis[0]*(basis[4]*basis[8]-basis[5]*basis[7])-
      basis[1]*(basis[3]*basis[8]-basis[5]*basis[6])+
      basis[2]*(basis[3]*basis[7]-basis[4]*basis[6])
  if(determinant<0) for(let row=0;row<3;row++) basis[row*3+2]*=-1
  return {eigenvalues:order.map(i=>a[i*3+i]), basis}
}
function metricCovariance(scales: ArrayLike<number>, quaternion: ArrayLike<number>, sx: number, sy: number) {
  const R=rotationFromQuaternion(...quaternion as [number,number,number,number])
  const scale=[sx,sy,1]
  const covariance=Array(9).fill(0)
  for(let i=0;i<3;i++) for(let j=0;j<3;j++) {
    let value=0
    for(let k=0;k<3;k++) value+=R[3*i+k]*R[3*j+k]*scales[k]*scales[k]
    covariance[3*i+j]=value*scale[i]*scale[j]
  }
  const {eigenvalues,basis}=diagonalize(covariance)
  return {
    scales:eigenvalues.map(v=>Math.sqrt(Math.max(1e-12,v))),
    quaternion:quaternionFromRotation(basis),
  }
}
export function decodeSharpGaussians(outputs: SharpOutputTensors, camera: SharpCamera): SharpGaussians {
  if (![camera.width,camera.height,camera.focalPx].every(n=>Number.isFinite(n)&&n>0)) {
    throw new Error('SHARP camera dimensions and focalPx must be positive finite numbers')
  }
  const threshold=camera.opacityThreshold ?? 0.1
  const max=camera.maxGaussians ?? 5_000_000
  if (!Number.isFinite(threshold) || !Number.isInteger(max) || max<1) {
    throw new Error('Invalid SHARP opacity threshold or maxGaussians')
  }
  const means=tensorValues(outputs.mean_vectors_ndc,'mean_vectors_ndc',3)
  const scales=tensorValues(outputs.singular_values_ndc,'singular_values_ndc',3)
  const quats=tensorValues(outputs.quaternions_ndc,'quaternions_ndc',4)
  const colors=tensorValues(outputs.colors,'colors',3)
  const opacities=tensorValues(outputs.opacities,'opacities',1)
  const totalCount=opacities.length
  if(!totalCount || means.length!==totalCount*3 || scales.length!==totalCount*3 ||
     colors.length!==totalCount*3 || quats.length!==totalCount*4) {
    throw new Error('SHARP output count mismatch')
  }
  const selected=Array.from({length:totalCount},(_,i)=>i).filter(i=>opacities[i]>=threshold)
  if(selected.length>max){
    selected.sort((a,b)=>opacities[b]-opacities[a])
    selected.length=max
    selected.sort((a,b)=>a-b)
  }
  const out:SharpGaussians={
    count:selected.length,totalCount,
    means:new Float32Array(selected.length*3),
    scales:new Float32Array(selected.length*3),
    quaternions:new Float32Array(selected.length*4),
    colors:new Float32Array(selected.length*3),
    opacities:new Float32Array(selected.length),
    camera:{width:camera.width,height:camera.height,focalPx:camera.focalPx},
  }
  const sx=camera.width/(2*camera.focalPx),sy=camera.height/(2*camera.focalPx)
  for(let i=0;i<selected.length;i++){
    const j=selected[i], a=j*3, b=i*3, q=j*4, p=i*4
    out.means.set([means[a]*sx,means[a+1]*sy,means[a+2]],b)
    const values=metricCovariance(scales.subarray(a,a+3),quats.subarray(q,q+4),sx,sy)
    out.scales.set(values.scales,b)
    out.quaternions.set(values.quaternion,p)
    out.colors.set(colors.subarray(a,a+3),b)
    out.opacities[i]=opacities[j]
  }
  return out
}
