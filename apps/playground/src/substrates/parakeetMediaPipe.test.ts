import { describe, expect, it } from 'vitest'
import {
  PARAKEET_MODEL_FILENAME,
  PARAKEET_MODEL_REVISION,
  PARAKEET_MODEL_URL,
  PARAKEET_WASM_URL,
  combineParakeetSegments,
  parakeetTranscript,
  validateParakeetAudio,
  validateParakeetModelFile,
} from './parakeetMediaPipe'

describe('opt-in MediaPipe Parakeet browser ASR', () => {
  it('pins model and runtime assets independently of caller audio', () => {
    expect(PARAKEET_MODEL_FILENAME).toBe('parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm')
    expect(PARAKEET_MODEL_REVISION).toMatch(/^[a-f0-9]{40}$/)
    expect(PARAKEET_MODEL_URL).toContain('/resolve/' + PARAKEET_MODEL_REVISION + '/' + PARAKEET_MODEL_FILENAME)
    expect(PARAKEET_WASM_URL).toContain('@mediapipe/tasks-audio@1.2.2/wasm')
  })

  it('allows only nonempty finite normalized PCM for transcription', () => {
    expect(() => validateParakeetAudio(new Float32Array())).toThrow(/empty/)
    expect(() => validateParakeetAudio(Float32Array.of(0.1, Number.NaN))).toThrow(/finite/)
    expect(() => validateParakeetAudio(Float32Array.of(-1.01))).toThrow(/normalized/)
    expect(() => validateParakeetAudio(Float32Array.of(-1, 0, 1))).not.toThrow()
  })

  it('replaces overlapping interim hypotheses with finalized segments without duplicating text', () => {
    const interim = { transcript: 'hello wor', isFinal: false, fromTimestampMs: 0, toTimestampMs: 1800 }
    const final = { transcript: 'Hello world.', isFinal: true, fromTimestampMs: 0, toTimestampMs: 2000 }
    const later = { transcript: 'Next sentence', isFinal: false, fromTimestampMs: 2000, toTimestampMs: 2600 }
    const first = combineParakeetSegments([], [interim])
    const second = combineParakeetSegments(first, [final, later])
    expect(parakeetTranscript(second)).toBe('Hello world. Next sentence')
    expect(parakeetTranscript(combineParakeetSegments(second, [final]))).toBe('Hello world. Next sentence')
    expect(parakeetTranscript(combineParakeetSegments(second, [interim]))).toBe('Hello world. Next sentence')
  })

  it('rejects incorrect, empty, and mislabeled local model files', () => {
    expect(() => validateParakeetModelFile({ name: 'fake.bin', size: 615000000 })).toThrow(/litertlm/)
    expect(() => validateParakeetModelFile({ name: 'speech.litertlm', size: 0 })).toThrow(/empty/)
    expect(() => validateParakeetModelFile({ name: 'speech.litertlm', size: 615000000 })).not.toThrow()
  })
})
