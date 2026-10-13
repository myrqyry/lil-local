import {describe, expect, it} from 'vitest'
import type {ManagedLiteRtRuntime, ManagedLiteRtRuntimeContext} from '@litert-playground/runtime-litert'
import {estimateMoGeGeometry, MOGE_GPU_URL, MOGE_CPU_URL, releaseMoGeModels} from './mogeGeometryService'
import {MOGE_PIXELS} from './mogeGeometry'

type MockTensor = {delete: () => void; moveTo: (backend: string) => Promise<MockTensor>; value: Float32Array}
const fixture = () => {
  const points = new Float32Array(MOGE_PIXELS * 3)
  const normals = new Float32Array(MOGE_PIXELS * 3)
  const mask = new Float32Array(MOGE_PIXELS).fill(0.93)
  for (let i=0; i<MOGE_PIXELS; i++) { points[3*i+2]=1.0+i/MOGE_PIXELS; normals[3*i+2]=1 }
  const shapes = [[1,448,448,3],[1,1,1,1],[1,448,448,3],[1,448,448,1]]
  return [normals,new Float32Array([0.8]),points,mask].map((value,index)=>({name:'o'+index,shape:shapes[index],dtype:'float32',value}))
}
const image = () => ({width:1,height:1,data:new Uint8ClampedArray([210,90,30,255])} as ImageData)

function createMock() {
  const heads = fixture()
  const calls: string[] = []
  let released = 0
  const tensor = (value: Float32Array): MockTensor => ({
    value,delete:()=>{released++},
    moveTo:async (backend:string) => {calls.push('readback:'+backend);return tensor(value)},
  })
  const runtime = {
    loadModel:async (url:string) => {calls.push('load:'+url);return {
      getInputDetails:()=>[{shape:[1,3,448,448],dtype:'float32'}],
      getOutputDetails:()=>heads,
    }},
    createTensor:(values:Float32Array,shape:number[]) => {
      calls.push('input:'+shape.join('x')+':'+values[0]);return tensor(values)
    },
    predict:async (url:string) => {
      calls.push('predict:'+url);return heads.map((o)=>tensor(o.value))
    },
    readTensor:(value:MockTensor)=>value.value,
    disposeModel:(url:string)=>calls.push('dispose:'+url),
  } as unknown as ManagedLiteRtRuntime
  const context = {backend:'webgpu',liteRt:runtime} as ManagedLiteRtRuntimeContext
  return {context,runtime,calls, getReleased:()=>released}
}

describe('MoGe-2 model runner boundary',()=>{
 it('loads GPU weights through caller runtime and asynchronously reads back four heads',async()=>{
  const mock=createMock()
  const result=await estimateMoGeGeometry(mock.context,image(),{backend:'webgpu',frameId:'frame-A'})
  expect(result.backend).toBe('webgpu')
  expect(result.frameId).toBe('frame-A')
  expect(result.relativeDepth[0]).toBe(0)
  expect(result.validFraction).toBe(1)
  expect(mock.calls).toContain('load:'+MOGE_GPU_URL)
  expect(mock.calls).toContain('predict:'+MOGE_GPU_URL)
  expect(mock.calls.filter(x=>x==='readback:wasm')).toHaveLength(4)
  expect(mock.getReleased()).toBe(9) // one input, four source tensors, four moved views
 })
 it('uses fp32 weights when WASM is requested without GPU readback',async()=>{
  const mock=createMock()
  const result=await estimateMoGeGeometry(mock.context,image(),{backend:'wasm'})
  expect(result.backend).toBe('wasm')
  expect(mock.calls).toContain('load:'+MOGE_CPU_URL)
  expect(mock.calls.some(x=>x.startsWith('readback:'))).toBe(false)
  expect(mock.getReleased()).toBe(5)
 })
 it('refuses to infer if cancellation precedes model loading',async()=>{
  const mock=createMock()
  const controller=new AbortController()
  controller.abort()
  await expect(estimateMoGeGeometry(mock.context,image(),{signal:controller.signal})).rejects.toThrow()
  expect(mock.calls).toHaveLength(0)
 })
 it('only disposes MoGe weights, not the shared runtime',()=>{
  const mock=createMock()
  releaseMoGeModels(mock.runtime)
  expect(mock.calls).toEqual(['dispose:'+MOGE_GPU_URL,'dispose:'+MOGE_CPU_URL])
 })
})
