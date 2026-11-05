import { describe, expect, it, vi } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  PARAKEET_MODEL_URL, PARAKEET_MODEL_REVISION, PARAKEET_WASM_URL,
  YAMNET_MODEL, createYamNetSubstrate, createParakeetSubstrate,
  createMediaPipeTtsSubstrate, validateParakeetAudio, validateParakeetModelFile,
  combineParakeetSegments, parakeetTranscript, verifyYamNetModelBytes,
  mapYamNetResults, validateAudioClip, validateTtsText, validateTtsResult,
} from './index'

describe('reusable MediaPipe Audio contracts', () => {
  it('pins SDK, Parakeet revision, and the independently hashed official YAMNet file', () => {
    expect(PARAKEET_MODEL_REVISION).toMatch(/^[0-9a-f]{40}$/)
    expect(PARAKEET_MODEL_URL).toContain('/resolve/' + PARAKEET_MODEL_REVISION + '/')
    expect(PARAKEET_WASM_URL).toContain('@mediapipe/tasks-audio@1.2.2/wasm')
    expect(YAMNET_MODEL.bytes).toBe(4_126_810)
    expect(YAMNET_MODEL.sha256).toBe('4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf')
  })

  it('rejects mismatched byte lengths and SHA-256 without running the model', async () => {
    const samples = Uint8Array.from([1, 2, 3])
    const sha256 = [...new Uint8Array(await crypto.subtle.digest('SHA-256', samples))].map(v=>v.toString(16).padStart(2,'0')).join('')
    await expect(verifyYamNetModelBytes(samples,{bytes:3,sha256})).resolves.toBeUndefined()
    await expect(verifyYamNetModelBytes(samples,{bytes:4,sha256})).rejects.toThrow(/size/)
    await expect(verifyYamNetModelBytes(Uint8Array.of(1,2,4),{bytes:3,sha256})).rejects.toThrow(/SHA-256/)
  })

  it('keeps every YAMNet time-window and classification head with its real scores', () => {
    const mapped=mapYamNetResults([
      {timestampMs:0,classifications:[{headIndex:0,headName:'sounds',categories:[{categoryName:'Speech',score:.8,index:0,displayName:'Speech'}]}]},
      {timestampMs:960,classifications:[{headIndex:0,headName:'sounds',categories:[{categoryName:'Music',score:.7,index:1,displayName:'Music'}]}]},
    ])
    expect(mapped.kind).toBe('audio-classification')
    expect(mapped.windows.map(w=>w.timestampMs)).toEqual([0,960])
    expect(mapped.windows.map(w=>w.heads[0].categories[0].categoryName)).toEqual(['Speech','Music'])
    expect(mapped.windows[0].heads[0].categories[0].score).toBe(.8)
  })

  it('validates audio boundaries, sample rate, TTS input, and local ASR model file', () => {
    expect(()=>validateAudioClip({samples:new Float32Array(0),sampleRate:16000})).toThrow(/empty/)
    expect(()=>validateAudioClip({samples:Float32Array.of(.1),sampleRate:0})).toThrow(/sample rate/)
    expect(()=>validateAudioClip({samples:Float32Array.of(NaN),sampleRate:16000})).toThrow(/finite/)
    expect(()=>validateAudioClip({samples:Float32Array.of(1.1),sampleRate:16000})).toThrow(/normalized/)
    expect(()=>validateAudioClip({samples:Float32Array.of(-1,0,1),sampleRate:16000})).not.toThrow()
    expect(()=>validateParakeetAudio(new Float32Array(0))).toThrow(/empty/)
    expect(()=>validateParakeetModelFile({name:'bad.bin',size:12})).toThrow(/litertlm/)
    expect(()=>validateParakeetModelFile({name:'speech.litertlm',size:0})).toThrow(/empty/)
    expect(()=>validateTtsText('   ')).toThrow(/empty/)
    expect(validateTtsText(' hello ')).toBe('hello')
  })

  it('keeps finalized transcription segments when late interim updates arrive', () => {
    const interim={transcript:'hello wor',isFinal:false,fromTimestampMs:0,toTimestampMs:1800}
    const final={transcript:'Hello world.',isFinal:true,fromTimestampMs:0,toTimestampMs:2000}
    const next={transcript:'Next',isFinal:false,fromTimestampMs:2000,toTimestampMs:2600}
    const output=combineParakeetSegments(combineParakeetSegments([], [interim]),[final,next])
    expect(parakeetTranscript(output)).toBe('Hello world. Next')
    expect(parakeetTranscript(combineParakeetSegments(output,[interim]))).toBe('Hello world. Next')
  })

  it('fails closed on invalid TTS model options and native PCM output', () => {
    expect(()=>createMediaPipeTtsSubstrate({model:''})).toThrow(/explicit model/)
    expect(()=>createMediaPipeTtsSubstrate({model:new Uint8Array()})).toThrow(/empty/)
    expect(()=>createMediaPipeTtsSubstrate({model:'https://example.org/model.litertlm',speakingRate:5})).toThrow(/speakingRate/)
    expect(()=>validateTtsResult({samples:new Float32Array(),sampleRateHz:24000})).toThrow(/invalid PCM/)
    expect(()=>validateTtsResult({samples:Float32Array.of(NaN),sampleRateHz:24000})).toThrow(/finite/)
    expect(()=>validateTtsResult({samples:Float32Array.of(1.2),sampleRateHz:24000})).toThrow(/normalized/)
    expect(validateTtsResult({samples:Float32Array.of(-1,0,1),sampleRateHz:24000})).toMatchObject({kind:'audio',channels:1,sampleRateHz:24000})
  })

  it('constructs three cold real substrate descriptors and never fetches during discovery', () => {
    const fetchSpy=vi.spyOn(globalThis,'fetch')
    const yamnet=new BrowserSubstrateRunner(createYamNetSubstrate())
    const asr=new BrowserSubstrateRunner(createParakeetSubstrate())
    const tts=new BrowserSubstrateRunner(createMediaPipeTtsSubstrate({model:new Uint8Array([1,2,3])}))
    expect([yamnet.status,asr.status,tts.status]).toEqual(['idle','idle','idle'])
    expect(yamnet.substrate.capabilities).toContain('audio-classification')
    expect(asr.substrate.capabilities).toContain('speech-to-text')
    expect(tts.substrate.capabilities).toContain('text-to-speech')
    expect(fetchSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
  })

  it('pre-aborted loads do not start any engine or weights', async () => {
    const abort=new AbortController()
    abort.abort()
    const original=globalThis.fetch
    const fetchSpy=vi.fn()
    globalThis.fetch=fetchSpy as typeof fetch
    try {
      const yamnet=new BrowserSubstrateRunner(createYamNetSubstrate())
      const asr=new BrowserSubstrateRunner(createParakeetSubstrate())
      const tts=new BrowserSubstrateRunner(createMediaPipeTtsSubstrate({model:new Blob(['x'])}))
      for (const runner of [yamnet,asr,tts]) {
        await expect(runner.load(undefined,abort.signal)).rejects.toMatchObject({name:'AbortError'})
        expect(runner.status).toBe('idle')
      }
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {globalThis.fetch=original}
  })
})
