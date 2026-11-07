import { createMediaPipeTtsEngine, validateTtsResult } from '../../packages/audio-mediapipe/src/index'

declare global {
  interface Window {
    runLilLocalTtsQualification: () => Promise<unknown>
  }
}

window.runLilLocalTtsQualification = async () => {
  const status = document.querySelector('#status')!
  const result = document.querySelector('#result')!
  status.textContent = 'loading'
  let engine: Awaited<ReturnType<typeof createMediaPipeTtsEngine>> | undefined
  try {
    engine = await createMediaPipeTtsEngine({
      model: '/__tts_assets/phonon_int8.litertlm',
      wasmBase: '/__tts_assets/wasm',
      delegate: 'CPU',
      defaultVoiceId: 'freya',
    })
    status.textContent = 'synthesizing'
    const voices = await engine.listVoices()
    const start = performance.now()
    const output = validateTtsResult(await engine.synthesize('Hello from Lil Local. This is real on device speech.', {voiceId: 'freya'}))
    const inferenceMs = performance.now() - start
    let energy = 0, peak = 0
    for (const sample of output.samples) {
      energy += sample * sample
      peak = Math.max(peak, Math.abs(sample))
    }
    const stream = await engine.synthesizeStream('The first clause is streamed. The second clause follows.', {voiceId: 'freya'})
    let streamedSamples = 0
    let streamedChunks = 0
    let sawFinalChunk = false
    try {
      for await (const chunk of stream) {
        validateTtsResult(chunk)
        if (chunk.chunkIndex !== streamedChunks) throw new Error('Streaming chunk index out of order')
        streamedChunks += 1
        streamedSamples += chunk.samples.length
        sawFinalChunk = chunk.isFinal
      }
    } finally {
      stream.close()
    }
    const switched = validateTtsResult(await engine.synthesize('Bonjour, je suis une autre voix.', {voiceId: 'claire'}))
    let switchedPeak = 0
    for (const sample of switched.samples) switchedPeak = Math.max(switchedPeak, Math.abs(sample))
    const report = {
      switchedSamples: switched.samples.length,
      switchedPeak,
      streamedSamples,
      streamedChunks,
      sawFinalChunk,
      voiceIds: voices.map(voice => typeof voice === 'string' ? voice : JSON.stringify(voice)),
      sampleRateHz: output.sampleRateHz,
      samples: output.samples.length,
      durationSeconds: output.samples.length / output.sampleRateHz,
      peak,
      rms: Math.sqrt(energy / output.samples.length),
      inferenceMs,
      wordTimestampCount: output.wordTimestamps?.length ?? 0,
    }
    status.textContent = 'complete'
    result.textContent = JSON.stringify(report)
    return report
  } catch (error) {
    status.textContent = 'failed'
    result.textContent = String(error)
    throw error
  } finally {
    engine?.close()
  }
}
