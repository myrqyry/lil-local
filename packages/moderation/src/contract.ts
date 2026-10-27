/**
 * Content-safety guard contract: pure input/output validation and score decoding.
 *
 * The model is `onnx-community/content-safety-guard-small-ONNX` — an Apache-2.0
 * ModernBERT multilingual safety classifier (base: Horizon-Labs/content-safety-guard-small).
 * It flags unsafe user prompts and unsafe model responses, optionally judging a
 * (prompt, response) pair in context, and emits an overall `unsafe` probability
 * plus independent per-harm-category sigmoids.
 */

/**
 * Immutable source revision observed from the Hugging Face Hub on 2026-10-26.
 * This is a source pin, NOT a verified weight checksum or a successful browser
 * inference receipt. Nothing downloads until an explicit load().
 */
export const MODERATION_MODEL_REVISION = '74e953e4b14199e152899135cb176a0998135baf'
export const MODERATION_MODEL_ID = 'onnx-community/content-safety-guard-small-ONNX'

/** The independent per-harm-category sigmoids the model emits, in card order. */
export const MODERATION_CATEGORIES = [
  'hate',
  'harassment',
  'violence',
  'weapons',
  'sexual',
  'sexual_minors',
  'self_harm',
  'criminal_planning',
  'drugs',
  'privacy',
  'profanity',
  'fraud_manipulation',
  'misinformation',
  'unauthorized_advice',
  'other',
] as const

export type ModerationCategory = (typeof MODERATION_CATEGORIES)[number]

export interface ModerationInput {
  /** The text to judge. For a model response, supply the originating prompt as `textPair`. */
  text: string
  /**
   * Optional second segment. When present it is judged *together with* `text`
   * (a prompt/response pair), so a response is scored in the context of its prompt.
   */
  textPair?: string
}

export interface ModerationResult {
  kind: 'moderation'
  /** Overall unsafe probability (independent sigmoid). The card's decision rule flags at >= 0.5. */
  unsafe: number
  /** Per-harm-category probabilities (independent sigmoids). Meaningful only for flagged text. */
  categories: Record<ModerationCategory, number>
}

/** One {label, score} entry as returned by the Transformers.js text-classification pipeline. */
interface LabelScore {
  label: string
  score: number
}

function normalizeLabelScores(raw: unknown): LabelScore[] {
  if (!Array.isArray(raw)) {
    throw new Error('Content-safety-guard returned an unexpected output shape')
  }
  // A single input with top_k: null returns a one-item batch: [[{label, score}, ...]].
  // Accept a flat label list too, so a caller feeding one text need not wrap it.
  const first: unknown = raw[0]
  return (Array.isArray(first) ? first : raw) as LabelScore[]
}

/**
 * Decode the pipeline's label scores into the structured moderation result.
 * Fails closed: a missing `unsafe` score or any missing category, or a
 * non-finite score, is an ABI/qualification error, never a fabricated result.
 */
export function decodeModerationScores(raw: unknown): Pick<ModerationResult, 'unsafe' | 'categories'> {
  const entries = normalizeLabelScores(raw)
  const byLabel = new Map<string, number>()
  for (const entry of entries) {
    if (entry === null || typeof entry !== 'object') {
      throw new Error('Content-safety-guard returned a malformed label score')
    }
    const { label, score } = entry as LabelScore
    if (typeof label !== 'string' || typeof score !== 'number' || !Number.isFinite(score)) {
      throw new Error('Content-safety-guard returned an invalid label score')
    }
    byLabel.set(label, score)
  }
  if (!byLabel.has('unsafe')) {
    throw new Error('Content-safety-guard output is missing the unsafe score')
  }
  const categories = {} as Record<ModerationCategory, number>
  for (const category of MODERATION_CATEGORIES) {
    const score = byLabel.get(category)
    if (score === undefined) {
      throw new Error(`Content-safety-guard output is missing the ${category} score`)
    }
    categories[category] = score
  }
  return { unsafe: byLabel.get('unsafe') as number, categories }
}

/** Validate a moderation input and normalize it; never infers on empty text. */
export function validateModerationInput(input: ModerationInput): ModerationInput {
  if (typeof input?.text !== 'string' || !input.text.trim()) {
    throw new Error('Moderation text must be nonempty')
  }
  if (input.textPair !== undefined && typeof input.textPair !== 'string') {
    throw new Error('Moderation textPair must be a string when provided')
  }
  return input.textPair === undefined
    ? { text: input.text }
    : { text: input.text, textPair: input.textPair }
}
