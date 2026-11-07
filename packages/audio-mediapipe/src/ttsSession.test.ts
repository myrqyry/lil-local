import { describe, expect, it, vi } from 'vitest'
import type { TextToSpeech } from '@mediapipe/tasks-audio'
import { createMediaPipeTtsStreamSession } from './ttsSession'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function fixture() {
  const read = deferred<{
    samples: Float32Array
    sampleRateHz: number
    chunkIndex: number
    isFinal: boolean
    wordTimestamps?: Array<{ word: string, startChar: number, endChar: number, startSec: number, endSec: number }>
  } | null>()
  const nativeStream = {
    readNext: vi.fn(() => read.promise),
    cancel: vi.fn(),
    close: vi.fn(),
  }
  const engine = {
    listVoices: vi.fn(async () => [{ voiceId: 'freya', displayName: 'Freya', locale: 'en-US' }]),
    synthesizeStream: vi.fn(async () => nativeStream),
    cancel: vi.fn(),
    close: vi.fn(),
  }
  return { engine, nativeStream, read, session: createMediaPipeTtsStreamSession(engine as unknown as TextToSpeech) }
}

describe('MediaPipe TTS consumer streaming session', () => {
  it('streams copied, ordered PCM chunks and keeps the caller voice', async () => {
    const { engine, nativeStream, read, session } = fixture()
    expect(await session.listVoices()).toMatchObject([{ voiceId: 'freya', locale: 'en-US' }])
    const iterator = session.stream(' hello ', { voiceId: 'freya' })[Symbol.asyncIterator]()
    const next = iterator.next()
    await vi.waitFor(() => expect(engine.synthesizeStream).toHaveBeenCalledWith('hello', {voiceId:'freya',locale:undefined}))
    const samples = Float32Array.of(.25, -.25)
    read.resolve({ samples, sampleRateHz: 24000, chunkIndex: 0, isFinal: true, wordTimestamps: [
      {word:'hello',startChar:0,endChar:5,startSec:0,endSec:.5},
    ] })
    const received = await next
    expect(received.done).toBe(false)
    expect(received.value).toMatchObject({kind:'audio',channels:1,chunkIndex:0,isFinal:true,sampleRateHz:24000})
    expect(received.value.samples).toEqual(samples)
    expect(received.value.samples).not.toBe(samples)
    samples[0] = 0
    expect(received.value.samples[0]).toBe(.25)
    expect(await iterator.next()).toMatchObject({done:true})
    expect(nativeStream.close).toHaveBeenCalledTimes(1)
    session.close()
    expect(engine.close).toHaveBeenCalledTimes(1)
  })

  it('aborts active inference and never delivers late PCM', async () => {
    const { engine, nativeStream, read, session } = fixture()
    const controller = new AbortController()
    const iterator = session.stream('speech', {signal: controller.signal})[Symbol.asyncIterator]()
    const pending = iterator.next()
    await vi.waitFor(() => expect(nativeStream.readNext).toHaveBeenCalledTimes(1))
    controller.abort()
    read.resolve({samples:Float32Array.of(.5),sampleRateHz:24000,chunkIndex:0,isFinal:true})
    await expect(pending).rejects.toMatchObject({name:'AbortError'})
    expect(engine.cancel).toHaveBeenCalled()
    expect(nativeStream.cancel).toHaveBeenCalled()
    expect(nativeStream.close).toHaveBeenCalledTimes(1)
    session.close()
  })

  it('rejects concurrent streams without stealing the first stream', async () => {
    const { nativeStream, read, session } = fixture()
    const first = session.stream('first')[Symbol.asyncIterator]()
    const pending = first.next()
    await vi.waitFor(() => expect(nativeStream.readNext).toHaveBeenCalledTimes(1))
    const second = session.stream('second')[Symbol.asyncIterator]()
    await expect(second.next()).rejects.toThrow(/already active/)
    session.cancel()
    read.resolve(null)
    await expect(pending).rejects.toMatchObject({name:'AbortError'})
    session.close()
  })

  it('rejects invalid PCM and closes the stream on error', async () => {
    const { nativeStream, read, session } = fixture()
    const iterator = session.stream('speech')[Symbol.asyncIterator]()
    const pending = iterator.next()
    await vi.waitFor(() => expect(nativeStream.readNext).toHaveBeenCalledTimes(1))
    read.resolve({samples:Float32Array.of(NaN),sampleRateHz:24000,chunkIndex:0,isFinal:true})
    await expect(pending).rejects.toThrow(/non-finite/)
    expect(nativeStream.close).toHaveBeenCalledTimes(1)
    session.close()
  })



  it('cancels the native stream when a consumer exits early', async () => {
    const { engine, nativeStream, read, session } = fixture()
    const iterator = session.stream('speech')[Symbol.asyncIterator]()
    const pending = iterator.next()
    await vi.waitFor(() => expect(nativeStream.readNext).toHaveBeenCalledTimes(1))
    read.resolve({samples:Float32Array.of(.3),sampleRateHz:24000,chunkIndex:0,isFinal:false})
    expect((await pending).done).toBe(false)
    await iterator.return?.()
    expect(nativeStream.cancel).toHaveBeenCalledTimes(1)
    expect(engine.cancel).toHaveBeenCalledTimes(1)
    expect(nativeStream.close).toHaveBeenCalledTimes(1)
    session.close()
  })

  it('rejects new work after close and closes the native model only once', async () => {
    const { engine, session } = fixture()
    session.close()
    session.close()
    await expect(session.stream('later')[Symbol.asyncIterator]().next()).rejects.toThrow(/closed/)
    await expect(session.listVoices()).rejects.toThrow(/closed/)
    expect(engine.close).toHaveBeenCalledTimes(1)
  })
})
