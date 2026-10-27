import { describe, expect, it } from 'vitest'
import {
  MODERATION_CATEGORIES,
  MODERATION_MODEL_ID,
  MODERATION_MODEL_REVISION,
  decodeModerationScores,
  validateModerationInput,
} from './contract'

function fullScores(overrides: Partial<Record<string, number>> = {}) {
  return [
    { label: 'unsafe', score: overrides.unsafe ?? 0.7 },
    ...MODERATION_CATEGORIES.map(category => ({
      label: category,
      score: overrides[category] ?? 0.1,
    })),
  ]
}

describe('content-safety-guard contract', () => {
  it('pins the Hub revision as an immutable source, not a weight receipt', () => {
    expect(MODERATION_MODEL_ID).toBe('onnx-community/content-safety-guard-small-ONNX')
    expect(MODERATION_MODEL_REVISION).toMatch(/^[0-9a-f]{40}$/)
  })

  it('declares the 15 documented harm categories', () => {
    expect(MODERATION_CATEGORIES).toHaveLength(15)
    expect(MODERATION_CATEGORIES).toContain('hate')
    expect(MODERATION_CATEGORIES).toContain('sexual_minors')
    expect(MODERATION_CATEGORIES).toContain('unauthorized_advice')
  })

  it('decodes a single-input batch into unsafe plus every category', () => {
    const { unsafe, categories } = decodeModerationScores([fullScores({ unsafe: 0.9, violence: 0.4 })])
    expect(unsafe).toBe(0.9)
    expect(categories.violence).toBe(0.4)
    expect(Object.keys(categories).sort()).toEqual([...MODERATION_CATEGORIES].sort())
  })

  it('accepts a flat label list as well as a one-item batch', () => {
    const flat = decodeModerationScores(fullScores())
    const nested = decodeModerationScores([fullScores()])
    expect(flat).toEqual(nested)
  })

  it('fails closed when the unsafe score or any category is missing', () => {
    expect(() => decodeModerationScores([{ label: 'hate', score: 0.1 }]))
      .toThrow(/missing the unsafe score/)
    const withoutCategory = fullScores().filter(entry => entry.label !== 'privacy')
    expect(() => decodeModerationScores([withoutCategory]))
      .toThrow(/missing the privacy score/)
  })

  it('rejects non-finite and malformed scores rather than fabricating results', () => {
    expect(() => decodeModerationScores([fullScores({ hate: Number.NaN })]))
      .toThrow(/invalid label score/)
    expect(() => decodeModerationScores([fullScores({ hate: Number.POSITIVE_INFINITY })]))
      .toThrow(/invalid label score/)
    expect(() => decodeModerationScores('not-an-array')).toThrow(/unexpected output shape/)
    expect(() => decodeModerationScores([{ label: 'unsafe', score: 0.5 }, null]))
      .toThrow(/malformed label score/)
  })

  it('requires nonempty text and types the optional text pair', () => {
    expect(() => validateModerationInput({ text: '   ' })).toThrow(/nonempty/)
    expect(() => validateModerationInput({ text: 'ok', textPair: 42 as never })).toThrow(/textPair/)
    expect(validateModerationInput({ text: 'prompt', textPair: 'response' }))
      .toEqual({ text: 'prompt', textPair: 'response' })
    expect(validateModerationInput({ text: 'solo' })).toEqual({ text: 'solo' })
  })
})
