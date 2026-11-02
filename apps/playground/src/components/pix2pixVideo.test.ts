import { describe, expect, it } from 'vitest'
import { buildFrameTimes, buildPreviewFrameTimes, chooseVideoExportMimeType, runFrameSequence } from './pix2pixVideo'

describe('buildPreviewFrameTimes', () => {
  it('samples a bounded two-second preview at the requested fps', () => {
    const times = buildPreviewFrameTimes(10, 8, 2)

    expect(times).toHaveLength(16)
    expect(times[0]).toBe(0)
    expect(times[times.length - 1]).toBeCloseTo(1.875)
  })

  it('never schedules timestamps beyond a short source clip', () => {
    expect(buildPreviewFrameTimes(0.3, 8, 2)).toEqual([0, 0.125, 0.25])
  })
})

describe('buildFrameTimes', () => {
  it('samples the full source duration at the requested fps', () => {
    const times = buildFrameTimes(2.1, 8)

    expect(times).toHaveLength(17)
    expect(times[0]).toBe(0)
    expect(times[times.length - 1]).toBe(2)
  })
})

describe('chooseVideoExportMimeType', () => {
  it('prefers VP9, then VP8, then generic WebM', () => {
    expect(chooseVideoExportMimeType(type => type.includes('vp9'))).toBe('video/webm;codecs=vp9')
    expect(chooseVideoExportMimeType(type => type.includes('vp8'))).toBe('video/webm;codecs=vp8')
    expect(chooseVideoExportMimeType(type => type === 'video/webm')).toBe('video/webm')
  })

  it('fails rather than silently choosing an unsupported recorder format', () => {
    expect(() => chooseVideoExportMimeType(() => false)).toThrow(/WebM recording is unavailable/)
  })
})

describe('runFrameSequence', () => {
  it('processes frames strictly one at a time and reports progress in order', async () => {
    let active = 0
    let maxActive = 0
    const processed: number[] = []
    const progress: number[] = []

    const count = await runFrameSequence({
      times: [0, 0.125, 0.25],
      signal: new AbortController().signal,
      processFrame: async time => {
        active += 1
        maxActive = Math.max(maxActive, active)
        await Promise.resolve()
        processed.push(time)
        active -= 1
      },
      onProgress: ({ completed }) => progress.push(completed),
    })

    expect(count).toBe(3)
    expect(processed).toEqual([0, 0.125, 0.25])
    expect(progress).toEqual([1, 2, 3])
    expect(maxActive).toBe(1)
  })

  it('stops before starting another frame after cancellation', async () => {
    const controller = new AbortController()
    const processed: number[] = []

    await expect(runFrameSequence({
      times: [0, 0.125, 0.25],
      signal: controller.signal,
      processFrame: async time => {
        processed.push(time)
        controller.abort()
      },
    })).rejects.toMatchObject({ name: 'AbortError' })

    expect(processed).toEqual([0])
  })
})

describe('video export timestamp contract', () => {
  it('uses source frame times independent of processing wall time', async () => {
    const { frameDurationSeconds } = await import('./pix2pixVideo')
    expect(frameDurationSeconds(0, 8, 1)).toBeCloseTo(0.125)
    expect(frameDurationSeconds(0.875, 8, 1)).toBeCloseTo(0.125)
    expect(frameDurationSeconds(0.875, 8, 0.9)).toBeCloseTo(0.025)
  })

  it('rejects an unbounded export before generating large time arrays', async () => {
    const { buildFrameTimes } = await import('./pix2pixVideo')
    expect(() => buildFrameTimes(3600, 8)).toThrow(/maximum/)
  })
})
