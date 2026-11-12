import { describe, expect, it, vi } from 'vitest'
import type { AudioTranscriber } from '@mediapipe/tasks-audio'
import { createMediaPipeAsrStreamSession } from './asrSession'

function fakeTranscriber() {
  const engine = {
    transcribeStream: vi.fn(),
    finishStream: vi.fn(),
    close: vi.fn(),
  }
  const session = createMediaPipeAsrStreamSession(engine as unknown as AudioTranscriber)
  return { engine, session }
}

const interim = {
  transcript: 'wait for',
  fromTimestampMs: 0,
  toTimestampMs: 1500,
  isFinal: false,
}
const final = {
  transcript: 'Wait for it!',
  fromTimestampMs: 0,
  toTimestampMs: 1700,
  isFinal: true,
}

describe('MediaPipe ASR stream session for external consumers', () => {
  it('serializes copied PCM, reconciles final segments, and flushes only after every chunk', async () => {
    const { engine, session } = fakeTranscriber()
    let releaseFirst!: (value: unknown) => void
    engine.transcribeStream
      .mockImplementationOnce(() => new Promise(resolve => { releaseFirst = resolve }))
      .mockResolvedValueOnce({ text: 'Wait for it!', segments: [final] })
    engine.finishStream.mockResolvedValue({ text: 'Wait for it!', segments: [final] })

    const samples = Float32Array.of(0.2, -0.2)
    const first = session.push(samples)
    samples[0] = 1
    const second = session.push(Float32Array.of(0.3))
    const ending = session.finish()
    await Promise.resolve() // let the first serialized native call start

    expect(engine.transcribeStream).toHaveBeenCalledTimes(1)
    expect([...engine.transcribeStream.mock.calls[0][0]]).toEqual([0.2, -0.2].map(Math.fround))
    expect(engine.finishStream).not.toHaveBeenCalled()

    releaseFirst({ text: 'wait for', segments: [interim] })
    const firstResult = await first
    expect(firstResult.text).toBe('wait for')
    const secondResult = await second
    expect(secondResult.text).toBe('Wait for it!')
    expect(secondResult.segments).toEqual([final])
    const finalResult = await ending
    expect(finalResult.text).toBe('Wait for it!')
    expect(finalResult.segments).toEqual([final])
    expect(engine.transcribeStream).toHaveBeenCalledTimes(2)
    expect(engine.finishStream).toHaveBeenCalledOnce()
    await session.close()
    expect(engine.close).toHaveBeenCalledOnce()
  })

  it('rejects malformed PCM without delivering it and refuses more chunks after finish', async () => {
    const { engine, session } = fakeTranscriber()
    engine.finishStream.mockResolvedValue({ text: '', segments: [] })
    await expect(session.push(Float32Array.of(Number.NaN))).rejects.toThrow(/finite/)
    expect(engine.transcribeStream).not.toHaveBeenCalled()
    await session.finish()
    await expect(session.push(Float32Array.of(0.5))).rejects.toThrow(/finished|finishing/i)
    await session.close()
  })

  it('invalidates late in-flight results when closed and closes the native engine after work settles', async () => {
    const { engine, session } = fakeTranscriber()
    let release!: (value: unknown) => void
    engine.transcribeStream.mockImplementation(() => new Promise(resolve => { release = resolve }))

    const pending = session.push(Float32Array.of(0.5))
    const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    await Promise.resolve() // make sure native work is in flight before closing
    const closed = session.close()
    expect(engine.close).not.toHaveBeenCalled()
    release({ text: 'discard', segments: [final] })
    await rejected
    await closed
    expect(engine.close).toHaveBeenCalledOnce()
    await session.close()
    expect(engine.close).toHaveBeenCalledOnce()
    await expect(session.push(Float32Array.of(0))).rejects.toThrow(/closed/)
  })

  it('allows an existing playground engine to survive a completed recording session', async () => {
    const engine = { transcribeStream: vi.fn(), finishStream: vi.fn(), close: vi.fn() }
    engine.transcribeStream.mockResolvedValue({ text: 'Hello', segments: [{ transcript: 'Hello', fromTimestampMs: 0, toTimestampMs: 600, isFinal: true }] })
    engine.finishStream.mockResolvedValue({ text: 'Hello', segments: [] })
    const session = createMediaPipeAsrStreamSession(engine as unknown as AudioTranscriber, {
      closeEngineOnClose: false,
    })
    await session.push(Float32Array.of(0.5))
    expect((await session.finish()).text).toBe('Hello')
    await session.close()
    expect(engine.close).not.toHaveBeenCalled()
  })

  it('returns snapshots that cannot mutate the accumulated transcript', async () => {
    const { engine, session } = fakeTranscriber()
    engine.transcribeStream.mockResolvedValue({ text: 'Wait for it!', segments: [final] })
    engine.finishStream.mockResolvedValue({ text: 'Wait for it!', segments: [final] })
    const first = await session.push(Float32Array.of(0.5))
    first.segments[0].transcript = 'corrupted'
    first.segments.push({ ...interim })
    const last = await session.finish()
    expect(last.text).toBe('Wait for it!')
    expect(last.segments).toEqual([final])
    await session.close()
  })
})
