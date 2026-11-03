import { describe, expect, it, vi } from 'vitest'
import { SaliencySegmentationPipeline } from './pipeline'
import type { ManagedLiteRtRuntime } from '@lil-local/runtime-litert'

describe('SaliencySegmentationPipeline', () => {
  it('loads only when invoked, runs real tensor preprocessor/decoder with WASM selection, disposes model and tensors', async () => {
    const deletedInputs: string[] = []
    const deletedOutputs: string[] = []
    const output = new Float32Array(320*320)
    output[1] = .75
    const runtime = {
      loadModel: vi.fn(async () => ({
        getInputDetails: () => [{shape:[1,3,320,320],dtype:'float32'}],
        getOutputDetails: () => [{shape:[1,1,320,320],dtype:'float32'}],
      })),
      createTensor: vi.fn((data: Float32Array,shape:number[]) => ({data,shape,delete:()=>deletedInputs.push('ok')})),
      predict: vi.fn(async () => ({saliency: {data:output,delete:()=>deletedOutputs.push('ok')}})),
      readTensor: vi.fn((tensor:{data:Float32Array})=>tensor.data),
      getModelInfo: vi.fn(()=>({resolvedBackend:'wasm'})),
      disposeModel: vi.fn(),
    } as unknown as ManagedLiteRtRuntime
    const pipeline = new SaliencySegmentationPipeline()
    expect(runtime.loadModel).not.toHaveBeenCalled()
    await pipeline.load({liteRt:runtime,backend:'wasm',assets:{resolve:vi.fn()}} as never)
    expect(pipeline.status).toBe('ready')
    const input={width:2,height:1,data:new Uint8ClampedArray([255,0,0,255,0,0,255,255])}
    const mask=await pipeline.run(input)
    expect(mask).toMatchObject({kind:'foreground-mask',sourceWidth:2,sourceHeight:1,width:320,height:320,backend:'wasm'})
    expect(mask.alpha[1]).toBe(191)
    expect(runtime.predict).toHaveBeenCalledTimes(1)
    expect(vi.mocked(runtime.predict).mock.calls[0][2]).toMatchObject({accelerator:'wasm',label:'foreground-segmentation'})
    expect(deletedInputs).toHaveLength(1)
    expect(deletedOutputs).toHaveLength(1)
    await pipeline.dispose()
    expect(runtime.disposeModel).toHaveBeenCalledOnce()
  })

  it('recovers to ready when tensor construction or inference throws; never retains input tensors', async () => {
    const deleted: string[] = []
    const runtime = {
      loadModel: vi.fn(async() => ({
        getInputDetails: () => [{shape:[1,3,320,320], dtype:'float32'}],
        getOutputDetails: () => [{shape:[1,1,320,320], dtype:'float32'}],
      })),
      createTensor: vi.fn()
        .mockImplementationOnce(() => { throw new Error('tensor allocation failed') })
        .mockImplementationOnce(() => ({delete:()=>deleted.push('input')})),
      predict: vi.fn(async () => { throw new Error('inference failed') }),
      disposeModel: vi.fn(),
    } as unknown as ManagedLiteRtRuntime
    const pipeline = new SaliencySegmentationPipeline()
    await pipeline.load({liteRt:runtime,backend:'wasm',assets:{resolve:vi.fn()}} as never)
    const input={width:1,height:1,data:new Uint8ClampedArray([1,2,3,255])}
    await expect(pipeline.run(input)).rejects.toThrow('tensor allocation failed')
    expect(pipeline.status).toBe('ready')
    await expect(pipeline.run(input)).rejects.toThrow('inference failed')
    expect(pipeline.status).toBe('ready')
    expect(deleted).toEqual(['input'])
  })

  it('rejects misleading model signatures before inference', async () => {
    const pipeline = new SaliencySegmentationPipeline()
    const runtime = {
      loadModel: vi.fn(async()=>({getInputDetails:()=>[{shape:[1,3,256,256],dtype:'float32'}],
        getOutputDetails:()=>[{shape:[1,1,320,320],dtype:'float32'}]})),
      disposeModel:vi.fn(),
    } as unknown as ManagedLiteRtRuntime
    await expect(pipeline.load({liteRt:runtime,backend:'wasm',assets:{resolve:vi.fn()}} as never)).rejects.toThrow(/signature/)
    expect(pipeline.status).toBe('error')
  })
})
