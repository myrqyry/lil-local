import { describe, it, expect } from 'vitest'
import { buildSharpPly, decodeSharpGaussians, type SharpOutputTensors, SharpOnnxWebGpu } from './index'

function sample(): SharpOutputTensors {
  return {
    mean_vectors_ndc: { type: 'float32', dims: [1, 2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) },
    singular_values_ndc: { type: 'float32', dims: [1, 2, 3], data: new Float32Array([1, 1, 1, 2, 2, 2]) },
    quaternions_ndc: { type: 'float32', dims: [1, 2, 4], data: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]) },
    colors: { type: 'float32', dims: [1, 2, 3], data: new Float32Array([1, 0.5, 0, 0.5, 0.5, 0.5]) },
    opacities: { type: 'float32', dims: [1, 2, 1], data: new Float32Array([0.9, 0.1]) },
  }
}

describe('SHARP ONNX Gaussian ABI', () => {
  it('selects opacity threshold and transforms NDC Gaussian centers into metric space', () => {
    const result = decodeSharpGaussians(sample(), { width: 100, height: 200, focalPx: 50, opacityThreshold: 0.2 })
    expect(result.totalCount).toBe(2)
    expect(result.count).toBe(1)
    expect(Array.from(result.means)).toEqual([1, 4, 3])
    expect(Array.from(result.scales)).toEqual([2, 1, 1])
  })

  it('caps by opacity and retains output-index order', () => {
    const result = decodeSharpGaussians(sample(), { width: 100, height: 100, focalPx: 50, opacityThreshold: 0, maxGaussians: 1 })
    expect(result.count).toBe(1)
    expect(result.opacities[0]).toBeCloseTo(0.9)
  })

  it('rejects missing or malformed outputs rather than returning a convincing empty scene', () => {
    const tensors = sample()
    delete (tensors as Partial<SharpOutputTensors>).colors
    expect(() => decodeSharpGaussians(tensors, { width: 100, height: 100, focalPx: 30 })).toThrow(/colors/)
    const invalid = sample()
    invalid.colors = { type: 'float32', dims: [1, 1, 3], data: new Float32Array([1, 1, 1]) }
    expect(() => decodeSharpGaussians(invalid, { width: 100, height: 100, focalPx: 30 })).toThrow(/count|shape/i)
  })

  it('writes binary Gaussian PLY with SHARP camera metadata and correct length', () => {
    const gaussian = decodeSharpGaussians(sample(), { width: 100, height: 200, focalPx: 50, opacityThreshold: 0.2 })
    const ply = buildSharpPly(gaussian)
    const text = new TextDecoder().decode(ply)
    expect(text).toContain('element vertex 1')
    expect(text).toContain('element intrinsic 9')
    const headerEnd = new TextEncoder().encode(text.split('end_header\n')[0] + 'end_header\n').length
    expect(ply.byteLength).toBe(headerEnd + 56 + 64 + 36 + 8 + 8 + 8 + 1 + 3)
    const data = new DataView(ply.buffer, ply.byteOffset)
    expect(data.getFloat32(headerEnd, true)).toBeCloseTo(1)
    expect(data.getFloat32(headerEnd + 4, true)).toBeCloseTo(4)
  })
})

describe('SHARP runtime boundary', () => {
  it('does not fetch or import GPU runtime until explicit load and enforces WebGPU', async () => {
    const instance = new SharpOnnxWebGpu({ graphUrl: 'http://localhost/sharp.onnx', externalDataUrl: 'http://localhost/sharp.onnx.data' })
    expect(instance.ready).toBe(false)
    await expect(instance.run({ rgb: new Float32Array(0), width: 100, height: 100, focalPx: 50 })).rejects.toThrow(/not loaded/i)
    await instance.dispose()
  })

  it('refuses non-finite and unsupported dimensions', () => {
    expect(() => decodeSharpGaussians(sample(), { width: 0, height: 200, focalPx: 50 })).toThrow()
    expect(() => decodeSharpGaussians(sample(), { width: 100, height: 200, focalPx: Infinity })).toThrow()
  })
})


describe('Lil Local reusable substrate integration', () => {
  it('publishes image-to-3d capability with no eager weight loading', async () => {
    const { createSharpSubstrate } = await import('./index')
    const { BrowserSubstrateRunner } = await import('@lil-local/inference-core')
    const substrate = createSharpSubstrate({
      graphUrl: '/models/sharp_web_predictor.onnx',
      externalDataUrl: '/models/sharp_web_predictor.onnx.data',
    })
    expect(substrate.capabilities).toEqual(['image-to-3d'])
    expect(substrate.requestedBackends).toEqual(['webgpu'])
    const runner = new BrowserSubstrateRunner(substrate)
    expect(runner.status).toBe('idle')
    await runner.dispose()
  })
})


describe('NDC covariance preservation', () => {
  it('retains rotated anisotropic Gaussian covariance through unequal image scaling', () => {
    const t=sample()
    t.mean_vectors_ndc={type:'float32',dims:[1,1,3],data:new Float32Array([0,0,2])}
    t.singular_values_ndc={type:'float32',dims:[1,1,3],data:new Float32Array([2,1,0.5])}
    t.quaternions_ndc={type:'float32',dims:[1,1,4],data:new Float32Array([Math.cos(Math.PI/8),0,0,Math.sin(Math.PI/8)])}
    t.colors={type:'float32',dims:[1,1,3],data:new Float32Array([0.1,0.2,0.3])}
    t.opacities={type:'float32',dims:[1,1,1],data:new Float32Array([1])}
    const out=decodeSharpGaussians(t,{width:200,height:300,focalPx:50})
    const [w,x,y,z]=out.quaternions
    const R=[
      1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
      2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
      2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y),
    ]
    const C=Array(9).fill(0)
    for(let i=0;i<3;i++)for(let j=0;j<3;j++)
      for(let k=0;k<3;k++)C[i*3+j]+=R[i*3+k]*R[j*3+k]*out.scales[k]**2
    expect(C[0]).toBeCloseTo(10,4)
    expect(C[1]).toBeCloseTo(9,4)
    expect(C[4]).toBeCloseTo(22.5,4)
    expect(C[8]).toBeCloseTo(0.25,4)
  })
  it('decodes actual float16 storage bits rather than interpreting half bit patterns as numbers', () => {
    const t=sample()
    t.opacities={type:'float16',dims:[1,2,1],data:new Uint16Array([0x3800,0x0000])}
    const out=decodeSharpGaussians(t,{width:100,height:100,focalPx:50,opacityThreshold:0.2})
    expect(out.count).toBe(1)
    expect(out.opacities[0]).toBe(0.5)
  })
})
