/**
 * D1 Omni 600M text-only System-One question encoding.
 *
 * Reproduces the text framing of LiquidAI/d1-omni-600M/prompt.py.
 * The ONNX export's model_quantized graph consumes exactly one question per
 * batch row. This deliberately does not claim image/audio support.
 */
export type D1Question =
  | { type: 'choice'; instructions: string; criteria: Record<string, string> }
  | { type: 'noul'; instructions: string }

export type D1Tokenizer = {
  bos_token_id: number
  encode(text: string, options: { add_special_tokens: false }): number[]
  model: { convert_tokens_to_ids(tokens: string[]): number[] }
}

const DELIMITERS = {
  state: '<|reserved_7|>',
  question: '<|reserved_8|>',
  option: '<|reserved_9|>',
  optionEnd: '<|reserved_10|>',
  decide: '<|reserved_11|>',
  mask: '<|mask|>',
} as const

export const D1_TEMPERATURES = {
  'choice:2': 1.7465145587921143,
  'choice:3-5': 1.3998981714248657,
  'choice:6-10': 1.1751071214675903,
  'choice:11+': 1.372515082359314,
  'noul:2': 1.6663223505020142,
} as const

export function escapeD1SpecialTokens(text: string): string {
  return text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>')
}

export function validateD1Question(question: D1Question): void {
  if (!question.instructions.trim()) throw new Error('D1 question instructions are required')
  if (question.type === 'choice') {
    const options = Object.entries(question.criteria)
    if (options.length < 2 || options.length > 16) {
      throw new Error('D1 choice requires between 2 and 16 named options')
    }
    if (options.some(([key]) => !key.trim())) throw new Error('D1 option names cannot be empty')
  } else if (question.type !== 'noul') {
    throw new Error('Unsupported D1 question type')
  }
}

export function renderD1Options(question: D1Question): string[] {
  validateD1Question(question)
  if (question.type === 'noul') {
    return ['false: no, the statement does not hold', 'true: yes, the statement holds']
  }
  return Object.entries(question.criteria).map(([name, description]) =>
    description ? name + ': ' + description : name,
  )
}

function markerId(tokenizer: D1Tokenizer, token: string): number {
  const result = tokenizer.model.convert_tokens_to_ids([token])
  if (result.length !== 1 || !Number.isSafeInteger(result[0]) || result[0] < 0) {
    throw new Error('D1 tokenizer missing required token: ' + token)
  }
  return result[0]
}

export type D1Encoded = {
  ids: bigint[]
  markerPositions: bigint[]
  qtype: bigint
  optionLabels: string[]
}

/** Returns model inputs without downloading, inferring, or manufacturing scores. */
export function encodeD1Text(
  tokenizer: D1Tokenizer,
  state: string,
  question: D1Question,
  maxLength = 512,
): D1Encoded {
  if (!Number.isSafeInteger(maxLength) || maxLength < 96 || maxLength > 2048) {
    throw new Error('D1 text context must be 96..2048 tokens for this browser path')
  }
  if (!Number.isSafeInteger(tokenizer.bos_token_id) || tokenizer.bos_token_id < 0) {
    throw new Error('D1 tokenizer BOS token missing')
  }
  const options = renderD1Options(question)
  const enc = (value: string) =>
    tokenizer.encode(escapeD1SpecialTokens(value), { add_special_tokens: false })
  const q = markerId(tokenizer, DELIMITERS.question)
  const opt = markerId(tokenizer, DELIMITERS.option)
  const mask = markerId(tokenizer, DELIMITERS.mask)
  const end = markerId(tokenizer, DELIMITERS.optionEnd)
  const decide = markerId(tokenizer, DELIMITERS.decide)
  const stateId = markerId(tokenizer, DELIMITERS.state)

  const budget = Math.max(96, Math.min(options.length * 24 + 32, Math.floor(maxLength / 2)))
  const perOption = Math.max(2, Math.floor((budget - 3 * options.length) / options.length))
  const questionIds = [q, ...enc(question.instructions)].slice(0, Math.max(16, budget))
  const markers: number[] = []
  for (const optionText of options) {
    markers.push(questionIds.length + 1)
    questionIds.push(opt, mask, ...enc(' ' + optionText).slice(0, perOption), end)
  }
  questionIds.push(decide)

  const room = Math.max(0, maxLength - questionIds.length - 2)
  const stateIds = [stateId, ...enc(state).slice(0, room)]
  const ids = [tokenizer.bos_token_id, ...stateIds, ...questionIds]
  const markerPositions = markers.map(position => position + 1 + stateIds.length)
  if (markerPositions[markerPositions.length - 1] >= maxLength || ids.length > maxLength || ids[ids.length - 1] !== decide) {
    throw new Error('D1 question options exceed the context budget')
  }
  if (markerPositions.some(position => ids[position] !== mask)) {
    throw new Error('D1 question marker position mismatch')
  }
  return {
    ids: ids.map(BigInt),
    markerPositions: markerPositions.map(BigInt),
    qtype: question.type === 'choice' ? 0n : 2n,
    optionLabels: question.type === 'choice' ? Object.keys(question.criteria) : ['false', 'true'],
  }
}

export type D1Decision = {
  type: 'choice' | 'noul'
  label: string
  confidence: number
  probabilities: Record<string, number>
  yesProbability?: number
  rawLogits: number[]
  elapsedMs: number
  provider: 'onnxruntime-web'
  backend: 'webgpu'
  modelRevision: string
  calibrated: false
}

export function decodeD1Scores(
  logits: readonly number[],
  labels: readonly string[],
  questionType: D1Question['type'],
): { label: string; confidence: number; probabilities: Record<string, number>; yesProbability?: number } {
  if (logits.length !== labels.length || logits.length < 2 || logits.some(v => !Number.isFinite(v))) {
    throw new Error('D1 returned invalid or incomplete option logits')
  }
  const n = logits.length
  const group = n <= 2 ? '2' : n <= 5 ? '3-5' : n <= 10 ? '6-10' : '11+'
  const temperature = questionType === 'noul'
    ? D1_TEMPERATURES['noul:2']
    : D1_TEMPERATURES[('choice:' + group) as keyof typeof D1_TEMPERATURES]
  const scaled = logits.map(value => value / temperature)
  const peak = Math.max(...scaled)
  const weights = scaled.map(value => Math.exp(value - peak))
  const total = weights.reduce((acc, value) => acc + value, 0)
  if (!(total > 0) || !Number.isFinite(total)) throw new Error('D1 softmax failed')
  const probabilities = Object.fromEntries(labels.map((label, i) => [label, weights[i] / total]))
  const winningIndex = weights.indexOf(Math.max(...weights))
  // Model option order for noul is [false, true]. No probability reordering.
  return {
    label: labels[winningIndex],
    confidence: weights[winningIndex] / total,
    probabilities,
    ...(questionType === 'noul' ? { yesProbability: probabilities.true } : {}),
  }
}
