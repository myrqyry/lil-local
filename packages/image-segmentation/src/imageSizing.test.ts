import { describe, expect, it, vi } from 'vitest'
import { boundInferenceImage } from './imageSizing'

describe('bounded image resizing', () => {
  it('uses the NEW image returned by resize rather than discarding it', async () => {
    type Image = { width: number; height: number; resize(width: number, height: number): Promise<Image> }
    const resized: Image = { width: 512, height: 256, resize: vi.fn() }
    const resize = vi.fn(async () => resized)
    const original: Image = { width: 720, height: 360, resize }
    const result = await boundInferenceImage(original)
    expect(resize).toHaveBeenCalledWith(512, 256)
    expect(result).toBe(resized)
    expect(original.width).toBe(720)
  })

  it('preserves an image already within the inference limit', async () => {
    const original = { width: 256, height: 240, resize: vi.fn() }
    expect(await boundInferenceImage(original)).toBe(original)
    expect(original.resize).not.toHaveBeenCalled()
  })
})
