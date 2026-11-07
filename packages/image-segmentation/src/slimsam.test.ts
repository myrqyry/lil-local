import { describe, expect, it } from 'vitest'
import { SLIMSAM_MODEL, selectSlimSamPoint } from './index'

describe('small quantized point-guided segmentation', () => {
  it('pins the lightweight SlimSAM ONNX checkpoint and keeps it opt-in', () => {
    expect(SLIMSAM_MODEL).toMatchObject({
      repo: 'Xenova/slimsam-77-uniform',
      revision: '5850ab45f587c112167512ffef949107115e26a0',
      device: 'wasm',
      dtype: 'q8',
    })
    expect(SLIMSAM_MODEL.quantizedWeightBytes).toBe(13_785_975)
  })

  it('rejects a non-image before trying to fetch a model', async () => {
    await expect(selectSlimSamPoint(new Blob(['x'], {type:'text/plain'}), {x:0,y:0}))
      .rejects.toThrow(/supported/i)
  })
})
