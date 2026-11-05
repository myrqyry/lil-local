import {describe, expect, it, vi, beforeEach} from 'vitest'
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS, createInteractiveSegmenterSubstrate,
  copyInteractiveMask, validateInteractiveStrokes,
} from './index'

const sample = [
  {mode:'positive' as const, points:[{x:.3,y:.3},{x:.48,y:.38},{x:.57,y:.39}]},
]
const native={
  createFromOptions:vi.fn(),setImage:vi.fn(),segment:vi.fn(),close:vi.fn(),
}
vi.mock('@mediapipe/tasks-vision',()=>({
  FilesetResolver:{forVisionTasks:vi.fn().mockResolvedValue({})},
  InteractiveSegmenter:{createFromOptions:native.createFromOptions},
}))
beforeEach(()=>{
  vi.clearAllMocks()
  native.createFromOptions.mockResolvedValue({
    setImage:native.setImage,segment:native.segment,close:native.close,
  })
  native.segment.mockReturnValue({
    width:2,height:2,
    getAsFloat32Array:()=>new Float32Array([0,.1,.88,1]),
    close:vi.fn(),
  })
})
describe('MediaPipe MagicTouch V2 shared interactive strokes',()=>{
  it('pins exact original official model digest and no mutable latest',()=>{
    expect(MEDIAPIPE_VISION_MODELS.interactive.bytes).toBe(30_525_312)
    expect(MEDIAPIPE_VISION_MODELS.interactive.sha256)
      .toBe('38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03')
    expect(MEDIAPIPE_VISION_MODELS.interactive.url)
      .toContain('/interactive_segmenter_v2/magic_touch/int8/1/interactive_segmentation.task')
  })
  it('translates bounded normalized stroke modes and preserves all points',()=>{
    const input=[...sample,
      {mode:'negative' as const,points:[{x:.6,y:.6},{x:.8,y:.8}]},
      {mode:'lasso' as const,points:[{x:.1,y:.1},{x:.1,y:.5},{x:.5,y:.5}]},
    ]
    const strokes=validateInteractiveStrokes(input)
    expect(strokes.map(x=>x.brushMode)).toEqual([1,2,3])
    expect(strokes.every(x=>x.isCompleted)).toBe(true)
    expect(strokes[0].point).toEqual(sample[0].points)
    const originalX=input[0].points[0].x
    input[0].points[0].x=.9
    expect(strokes[0].point[0].x).toBe(originalX)
    input[0].points[0].x=originalX
  })
  it('rejects invalid, out-of-bounds, short, negative-only and overlarge inputs',()=>{
    expect(()=>validateInteractiveStrokes([])).toThrow(/stroke/i)
    expect(()=>validateInteractiveStrokes([{mode:'negative',points:sample[0].points}])).toThrow(/positive|lasso/i)
    expect(()=>validateInteractiveStrokes([{mode:'positive',points:[{x:.1,y:.1},{x:.101,y:.101}]}])).toThrow(/length/i)
    expect(()=>validateInteractiveStrokes([{mode:'positive',points:[{x:-.1,y:.1},{x:.3,y:.2}]}])).toThrow(/normalized/i)
    expect(()=>validateInteractiveStrokes([{mode:'wrong' as never,points:sample[0].points}])).toThrow(/mode/i)
    expect(()=>validateInteractiveStrokes(Array.from({length:34},()=>sample[0]))).toThrow(/too many/i)
    expect(()=>validateInteractiveStrokes([{mode:'positive',points:Array.from({length:257},(_,i)=>({x:i/300,y:.3}))}])).toThrow(/points/i)
  })
  it('copies float32 confidence masks, refuses invalid geometry and values',()=>{
    const src=new Float32Array([0,.25,.7,1])
    const mask=copyInteractiveMask({width:2,height:2,values:src})
    src[2]=0
    expect(mask.kind).toBe('interactive-segmentation')
    expect(mask.confidence[0]).toBe(0)
    expect(mask.confidence[1]).toBe(.25)
    expect(mask.confidence[2]).toBeCloseTo(.7)
    expect(mask.confidence[3]).toBe(1)
    expect(()=>copyInteractiveMask({width:2,height:2,values:new Float32Array(1)})).toThrow(/length/i)
    expect(()=>copyInteractiveMask({width:0,height:2,values:new Float32Array(0)})).toThrow(/dimensions/i)
    expect(()=>copyInteractiveMask({width:2,height:1,values:new Float32Array([.2,NaN])})).toThrow(/confidence/i)
  })
  it('stays cold, and aborted load skips model fetch',async()=>{
    const runner=new BrowserSubstrateRunner(createInteractiveSegmenterSubstrate())
    expect(runner.status).toBe('idle')
    expect(runner.substrate.capabilities).toContain('image-segmentation')
    const fetcher=vi.spyOn(globalThis,'fetch')
    const abort=new AbortController();abort.abort()
    try{
      await expect(runner.load(undefined,abort.signal)).rejects.toMatchObject({name:'AbortError'})
      expect(fetcher).not.toHaveBeenCalled()
    }finally{fetcher.mockRestore()}
  })
  it('requires an image and reuses one native encoder for repeated stroke refinements',async()=>{
    const model=new Uint8Array(30_525_312)
    // Only the SDK boundary/lifecycle is mocked here. Integrity checks
    // themselves have separate real-digest tests; browser smoke uses real bytes.
    const sha='38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03'
    const digest=vi.spyOn(globalThis.crypto.subtle,'digest').mockResolvedValue(
      Uint8Array.from(sha.match(/../g)!.map(x=>parseInt(x,16))).buffer as ArrayBuffer,
    )
    const maskClose=vi.fn()
    native.segment.mockImplementation(()=>({
      width:2,height:2,
      getAsFloat32Array:()=>new Float32Array([0,.1,.88,1]),
      close:maskClose,
    }))
    const runner=new BrowserSubstrateRunner(createInteractiveSegmenterSubstrate({model}))
    const image={} as ImageBitmap
    try{
      await expect(runner.run({kind:'segment',strokes:sample})).rejects.toThrow(/idle/)
      await runner.load()
      await expect(runner.run({kind:'segment',strokes:sample})).rejects.toThrow(/Set the image/)
      await expect(runner.run({kind:'set-image',image})).resolves.toEqual({kind:'image-ready'})
      const first=await runner.run({kind:'segment',strokes:sample})
      const second=await runner.run({kind:'segment',strokes:[
        ...sample,{mode:'negative',points:[{x:.6,y:.6},{x:.7,y:.65}]},
      ]})
      expect(first.kind).toBe('interactive-segmentation')
      expect(second.kind).toBe('interactive-segmentation')
      expect(native.createFromOptions).toHaveBeenCalledTimes(1)
      expect(native.setImage).toHaveBeenCalledTimes(1)
      expect(native.segment).toHaveBeenCalledTimes(2)
      expect(maskClose).toHaveBeenCalledTimes(2)
      await runner.run({kind:'set-image',image})
      expect(native.setImage).toHaveBeenCalledTimes(2)
    }finally{
      await runner.dispose()
      digest.mockRestore()
    }
    expect(native.close).toHaveBeenCalledTimes(1)
  })
})
