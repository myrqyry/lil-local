import { describe, expect, it } from 'vitest'
import { prepareSamPointPrompts, type SamPointPrompt } from './index'

describe('SAM 2.1 signed multi-point prompts', () => {
  it('maps include/exclude clicks to correctly shaped model processor arrays', () => {
    const points: SamPointPrompt[] = [
      { x: 190, y: 200, label: 1 },
      { x: 40, y: 60, label: 0 },
      { x: 380, y: 470, label: 1 },
    ]
    expect(prepareSamPointPrompts(points, 384, 480, 192, 240)).toEqual({
      input_points: [[[95,100], [20,30], [190,235]]],
      input_labels: [[1,0,1]],
    })
    expect(points[0].x).toBe(190)
  })
  it('prevents negative-only or malformed prompts from entering the model', () => {
    const valid = { x: 20, y: 30, label: 1 as const }
    expect(() => prepareSamPointPrompts([{ ...valid,label:0 }],384,480,384,480)).toThrow(/include/i)
    expect(() => prepareSamPointPrompts([],384,480,384,480)).toThrow(/include/i)
    expect(() => prepareSamPointPrompts([valid,{ x:3,y:3,label:2 } as unknown as SamPointPrompt],384,480,384,480)).toThrow(/label/i)
    expect(() => prepareSamPointPrompts([valid,{ x:-1,y:3,label:0 }],384,480,384,480)).toThrow(/outside/i)
    expect(() => prepareSamPointPrompts(Array(17).fill(valid),384,480,384,480)).toThrow(/16/i)
    expect(() => prepareSamPointPrompts([valid],0,480,384,480)).toThrow(/image/i)
  })
})
