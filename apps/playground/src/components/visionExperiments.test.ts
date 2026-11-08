import { describe, expect, it } from 'vitest'
import {
  appendExperiment, createVisionExperiment, renderSampledPixels, MAX_EXPERIMENTS,
} from './visionExperiments'

const defaults = {
  task: 'person',
  requestedBackend: 'wasm',
  image: { name: 'portrait.png', width: 2, height: 2, preview: 'data:image/png;base64,AAAA' },
  loadMs: 23.5,
  inferenceMs: 42.25,
  capturedAt: 500,
}

describe('Live Vision in-session experiment records', () => {
  it('owns a bounded float copy without retaining the model result', () => {
    const native = Float32Array.of(0.1, 0.4, 0.7, 0.9)
    const run = createVisionExperiment({
      ...defaults, id: 7,
      output: { kind: 'person-mask', width: 2, height: 2, confidence: native },
    })
    expect(run.id).toBe(7)
    expect(run.task).toBe('person')
    expect(run.requestedBackend).toBe('wasm')
    expect(run.loadMs).toBe(23.5)
    expect(run.inferenceMs).toBe(42.25)
    expect(run.raster?.kind).toBe('person-mask')
    expect(Array.from(run.raster!.values)).toEqual([0.1, 0.4, 0.7, 0.9].map(x => Math.fround(x)))
    native[0] = 0.95
    expect(run.raster?.values[0]).toBeCloseTo(0.1)
  })

  it('downsamples large frames to a bounded nearest-neighbor inspectable sample', () => {
    const pixels = Float32Array.of(
      0, 0.1, 0.2, 0.3,
      0.4, 0.5, 0.6, 0.7,
      0.8, 0.9, 1, 0.6,
      0.2, 0.4, 0.6, 0.8,
    )
    const entry = createVisionExperiment({
      ...defaults, id: 1,
      output: { kind: 'depth', width: 4, height: 4, data: pixels },
    }, 2)
    expect(entry.raster?.width).toBe(2)
    expect(entry.raster?.height).toBe(2)
    expect(Array.from(entry.raster!.values)).toEqual([0, Math.fround(0.2), Math.fround(0.8), 1])
    expect(entry.raster?.sourceWidth).toBe(4)
    expect(entry.raster?.sourceHeight).toBe(4)
  })

  it('re-decodes a stored person mask with a different threshold without inference', () => {
    const record = createVisionExperiment({
      ...defaults, id: 1,
      output: { kind: 'person-mask', width: 2, height: 2, confidence: Float32Array.of(0.1,0.4,0.7,0.9) },
    })
    const low = renderSampledPixels(record.raster!, { mode: 'cutoff', threshold: 0.3 })
    const high = renderSampledPixels(record.raster!, { mode: 'cutoff', threshold: 0.8 })
    expect([...low.filter((_, i) => i % 4 === 0)]).toEqual([0, 255, 255, 255])
    expect([...high.filter((_, i) => i % 4 === 0)]).toEqual([0, 0, 0, 255])
    expect([...renderSampledPixels(record.raster!, { mode: 'raw', threshold: 0.8 }).filter((_, i) => i % 4 === 0)]).toEqual([26, 102, 178, 229])
  })

  it('keeps only four newest results, ordered newest-first, without mutating earlier arrays', () => {
    let entries: ReturnType<typeof createVisionExperiment>[] = []
    for (let i = 1; i <= MAX_EXPERIMENTS + 2; i++) {
      const prior = entries
      entries = appendExperiment(entries, createVisionExperiment({
        ...defaults, id: i, output: { kind: 'image-classifications', heads: [] },
      }))
      expect(prior.length).toBeLessThanOrEqual(MAX_EXPERIMENTS)
    }
    expect(entries.map(e => e.id)).toEqual([6,5,4,3])
    expect(entries[0].summary).toContain('classification')
    expect(entries[0].raster).toBeNull()
  })

  it('rejects invalid numerical output and nonsensical timings rather than showing made-up statistics', () => {
    expect(() => createVisionExperiment({
      ...defaults, id: 1, output: {kind:'depth',width:2,height:2,data:Float32Array.of(0,1)},
    })).toThrow(/length/)
    expect(() => createVisionExperiment({
      ...defaults, id: 1, output: {kind:'person-mask',width:1,height:1,confidence:Float32Array.of(NaN)},
    })).toThrow(/finite/)
    expect(() => createVisionExperiment({
      ...defaults, id: 1, loadMs: -1, output: {kind:'image-classifications',heads:[]},
    })).toThrow(/timing/)
  })
})
