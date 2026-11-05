import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  MEDIAPIPE_VISION_MODELS, createImageClassifierSubstrate, mapImageClassifierResult,
} from './index'

describe('MediaPipe EfficientNet Lite0 image classifier', () => {
  it('pins the actually downloaded publisher asset', () => {
    expect(MEDIAPIPE_VISION_MODELS.classifier.bytes).toBe(18_582_189)
    expect(MEDIAPIPE_VISION_MODELS.classifier.sha256)
      .toBe('6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0')
    expect(MEDIAPIPE_VISION_MODELS.classifier.url)
      .toContain('image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite')
  })

  it('preserves all classifier heads, their indices and native labels/scores', () => {
    const mapped=mapImageClassifierResult({
      timestampMs: 1500,
      classifications: [
        {headIndex:0,headName:'species',categories:[
          {index:207,categoryName:'golden retriever',displayName:'Golden retriever',score:.92},
          {index:208,categoryName:'Labrador retriever',displayName:'Labrador retriever',score:.21},
        ]},
        {headIndex:1,headName:'background',categories:[
          {index:0,categoryName:'grass',displayName:'',score:.67},
        ]},
      ],
    })
    expect(mapped.kind).toBe('image-classifications')
    expect(mapped.timestampMs).toBe(1500)
    expect(mapped.heads.map(x=>[x.headIndex,x.headName])).toEqual([[0,'species'],[1,'background']])
    expect(mapped.heads[0].categories.map(x=>x.score)).toEqual([.92,.21])
    expect(mapped.heads[1].categories[0].index).toBe(0)
  })

  it('keeps empty results and absent timestamps intact', () => {
    expect(mapImageClassifierResult({classifications:[]})).toEqual({
      kind:'image-classifications',heads:[],
    })
  })

  it('rejects nonfinite scores or invalid probabilities without inventing output', () => {
    for(const score of [NaN,Infinity,-.1,1.1]) {
      expect(()=>mapImageClassifierResult({classifications:[
        {headIndex:0,headName:'main',categories:[
          {index:1,categoryName:'dog',displayName:'Dog',score},
        ]},
      ]})).toThrow(/score/)
    }
  })

  it('validates threshold and maximum categories before any download', () => {
    expect(()=>createImageClassifierSubstrate({maxResults:0})).toThrow(/maxResults/)
    expect(()=>createImageClassifierSubstrate({maxResults:2.7})).toThrow(/maxResults/)
    expect(()=>createImageClassifierSubstrate({maxResults:1001})).toThrow(/maxResults/)
    expect(()=>createImageClassifierSubstrate({scoreThreshold:-.01})).toThrow(/scoreThreshold/)
    expect(()=>createImageClassifierSubstrate({scoreThreshold:NaN})).toThrow(/scoreThreshold/)
    expect(()=>createImageClassifierSubstrate({scoreThreshold:1.01})).toThrow(/scoreThreshold/)
  })

  it('stays cold until opt-in load; aborted loads never fetch', async () => {
    const fetcher=vi.spyOn(globalThis,'fetch')
    try {
      const runner=new BrowserSubstrateRunner(createImageClassifierSubstrate({maxResults:3}))
      expect(runner.status).toBe('idle')
      expect(runner.substrate.capabilities).toContain('image-classification')
      expect(fetcher).not.toHaveBeenCalled()
      const abort=new AbortController(); abort.abort()
      await expect(runner.load(undefined,abort.signal))
        .rejects.toMatchObject({name:'AbortError'})
      expect(fetcher).not.toHaveBeenCalled()
    } finally { fetcher.mockRestore() }
  })
})
