import {describe,expect,it,vi} from 'vitest'
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {DEPTH_ANYTHING_V2,createDepthAnythingV2Substrate,onnxRuntimeLocalPaths} from './index'

describe('Opt-in quantized Depth Anything V2 browser substrate',()=>{
  it('pins the official model revision and uses the Q8 model on WASM',()=>{
    expect(DEPTH_ANYTHING_V2.id).toBe('onnx-community/depth-anything-v2-small')
    expect(DEPTH_ANYTHING_V2.revision).toBe('4472b7362082ad9968fee890ca0f1e5aca36b93d')
    expect(DEPTH_ANYTHING_V2.wasmDtype).toBe('q8')
  })
  it('self-hosts the matching runtime factory and wasm bytes',()=>{
    expect(onnxRuntimeLocalPaths('http://127.0.0.1:5173','/ort/')).toEqual({
      mjs:'http://127.0.0.1:5173/ort/ort-wasm-simd-threaded.asyncify.mjs',
      wasm:'http://127.0.0.1:5173/ort/ort-wasm-simd-threaded.asyncify.wasm',
    })
    expect(()=>onnxRuntimeLocalPaths('http://127.0.0.1:5173','https://cdn.jsdelivr.net/')).toThrow(/origin|local/i)
  })
  it('never loads model assets before explicit, non-aborted load',async()=>{
    const fetcher=vi.spyOn(globalThis,'fetch')
    try{
      const runner=new BrowserSubstrateRunner(createDepthAnythingV2Substrate({wasmBase:'/ort/'}))
      expect(runner.substrate.requestedBackends).toEqual(['wasm'])
      expect(runner.substrate.capabilities).toContain('image-depth')
      expect(fetcher).not.toHaveBeenCalled()
      const controller=new AbortController();controller.abort()
      await expect(runner.load(undefined,controller.signal)).rejects.toMatchObject({name:'AbortError'})
      expect(fetcher).not.toHaveBeenCalled()
    }finally{fetcher.mockRestore()}
  })
})
