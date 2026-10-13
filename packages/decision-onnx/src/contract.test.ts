import { describe, expect, it } from 'vitest'
import {
  decodeD1Scores, encodeD1Text, escapeD1SpecialTokens, renderD1Options,
  type D1Tokenizer,
} from './contract'

const special = new Map([
  ['<|reserved_7|>', 7], ['<|reserved_8|>', 8], ['<|reserved_9|>', 9],
  ['<|reserved_10|>', 10], ['<|reserved_11|>', 11], ['<|mask|>', 12],
])

const tokenizer: D1Tokenizer = {
  bos_token_id: 1,
  // Deterministic tokenizer substitute tests positions and budgeting, not model accuracy.
  encode: value => [...value].map(character => character.charCodeAt(0) + 100),
  model: { convert_tokens_to_ids: tokens => tokens.map(token => special.get(token) ?? -1) },
}

describe('D1 Omni text decision contract', () => {
  it('constructs a model-compatible question with exact mask positions', () => {
    const question = {
      type: 'choice' as const,
      instructions: 'Which animal?',
      criteria: { cat: 'Feline', dog: 'Canine' },
    }
    const result = encodeD1Text(tokenizer, 'Picture shows a cat', question, 512)
    const ids = result.ids.map(Number)
    expect(ids[0]).toBe(1)
    expect(ids[1]).toBe(7)
    expect(ids).toContain(8)
    expect(ids[ids.length - 1]).toBe(11)
    expect(result.qtype).toBe(0n)
    expect(result.markerPositions).toHaveLength(2)
    expect(result.markerPositions.every(position => ids[Number(position)] === 12)).toBe(true)
    expect(result.optionLabels).toEqual(['cat', 'dog'])
  })

  it('truncates long state before cutting option blocks', () => {
    const result = encodeD1Text(tokenizer, 'x'.repeat(1800), {
      type: 'choice', instructions: 'Choose', criteria: { one: 'First', two: 'Second' },
    }, 256)
    expect(result.ids.length).toBeLessThanOrEqual(256)
    expect(result.ids[result.ids.length - 1]).toBe(11n)
    expect(result.markerPositions.every(position => result.ids[Number(position)] === 12n)).toBe(true)
  })

  it('escapes caller-supplied reserved delimiters', () => {
    expect(escapeD1SpecialTokens('<|mask|> and <|reserved_8|>')).toBe('<¦mask¦> and <¦reserved_8¦>')
  })

  it('formats noul options in false then true order', () => {
    expect(renderD1Options({ type: 'noul', instructions: 'There is a cat' })).toEqual([
      'false: no, the statement does not hold',
      'true: yes, the statement holds',
    ])
    const result = encodeD1Text(tokenizer, 'A cat', { type: 'noul', instructions: 'Cat?' })
    expect(result.qtype).toBe(2n)
    expect(result.optionLabels).toEqual(['false', 'true'])
  })

  it('applies model temperature and preserves noul yes semantics', () => {
    const no = decodeD1Scores([4, 0], ['false', 'true'], 'noul')
    expect(no.label).toBe('false')
    expect(no.yesProbability).toBeLessThan(0.5)
    const yes = decodeD1Scores([0, 4], ['false', 'true'], 'noul')
    expect(yes.label).toBe('true')
    expect(yes.yesProbability).toBeGreaterThan(0.5)
    expect(yes.probabilities.true + yes.probabilities.false).toBeCloseTo(1)
  })

  it('rejects invalid options, context sizes and nonfinite model scores', () => {
    expect(() => encodeD1Text(tokenizer, '', {
      type: 'choice', instructions: 'Pick', criteria: { only: '' },
    })).toThrow(/2 and 16/)
    expect(() => encodeD1Text(tokenizer, '', { type: 'noul', instructions: 'hi' }, 64))
      .toThrow(/96..2048/)
    expect(() => decodeD1Scores([NaN, 0], ['no', 'yes'], 'noul'))
      .toThrow(/invalid/)
    expect(() => decodeD1Scores([1], ['only'], 'choice')).toThrow(/incomplete/)
  })

  it('fails closed when required model markers are absent', () => {
    const missing: D1Tokenizer = {
      ...tokenizer,
      model: { convert_tokens_to_ids: tokens => tokens.map(token => token === '<|mask|>' ? -1 : special.get(token) ?? -1) },
    }
    expect(() => encodeD1Text(missing, '', { type: 'noul', instructions: 'Yes?' }))
      .toThrow(/tokenizer missing required/)
  })
})
