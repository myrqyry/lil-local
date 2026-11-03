import { describe, expect, it } from 'vitest'
import { MODEL_REVISION, MODEL_ROOT, chooseNemotronBackend, validateAudioSamples } from './nemotronWeb'

describe('opt-in browser Nemotron ASR contract', () => {
  it('pins the ONNX INT4 model and never points the browser runtime at GGUF/JNI', () => {
    expect(MODEL_REVISION).toBe('8364d9e2dd9da23789b480bdbba9e423717e42ee')
    expect(MODEL_ROOT).toContain('/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/')
    expect(MODEL_ROOT).not.toContain('gguf')
  })

  it('requires explicit backend decisions without silently switching to cloud', () => {
    expect(chooseNemotronBackend(true, 'webgpu')).toBe('webgpu')
    expect(() => chooseNemotronBackend(false, 'webgpu')).toThrow(/WebGPU/)
    expect(chooseNemotronBackend(false, 'wasm')).toBe('wasm')
    expect(chooseNemotronBackend(false, 'auto')).toBe('wasm')
  })

  it('refuses empty, non-finite, or invalid PCM before scheduling inference', () => {
    expect(() => validateAudioSamples(new Float32Array())).toThrow(/empty/i)
    expect(() => validateAudioSamples(Float32Array.of(0, Number.NaN))).toThrow(/finite/i)
    expect(() => validateAudioSamples(Float32Array.of(.2, -.1))).not.toThrow()
  })
})
