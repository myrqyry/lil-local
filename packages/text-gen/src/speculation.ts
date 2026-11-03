export interface SuffixDraftCandidate {
  tokens: number[]
  matchOrder: number
  occurrences: number
}

export interface SuffixDraftOptions {
  maxOrder?: number
  maxDraftTokens?: number
  maxCandidates?: number
}

const DEFAULT_MAX_ORDER = 8
const DEFAULT_MAX_DRAFT_TOKENS = 8
const DEFAULT_MAX_CANDIDATES = 32

function positiveInteger(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError('speculation limits must be positive integers')
  }
  return value
}

function sameSlice(
  source: readonly number[],
  offset: number,
  suffix: readonly number[],
): boolean {
  for (let index = 0; index < suffix.length; index++) {
    if (source[offset + index] !== suffix[index]) return false
  }
  return true
}

/**
 * Collect exact suffix continuations from caller-owned token histories.
 *
 * This is intentionally the small reusable mechanism we can support today:
 * it does not claim to reproduce the paper's GPU parallel scan or weighted
 * context mixer. Higher-order matches dominate lower-order matches for the
 * same continuation, while repeated matches at that order increase evidence.
 */
export function collectSuffixDraftCandidates(
  current: readonly number[],
  sources: readonly (readonly number[])[],
  options: SuffixDraftOptions = {},
): SuffixDraftCandidate[] {
  const maxOrder = Math.min(
    current.length,
    positiveInteger(options.maxOrder, DEFAULT_MAX_ORDER),
  )
  const maxDraftTokens = positiveInteger(
    options.maxDraftTokens,
    DEFAULT_MAX_DRAFT_TOKENS,
  )
  const maxCandidates = positiveInteger(
    options.maxCandidates,
    DEFAULT_MAX_CANDIDATES,
  )

  if (maxOrder === 0 || sources.length === 0) return []

  const candidates = new Map<string, SuffixDraftCandidate>()

  for (let order = maxOrder; order >= 1; order--) {
    const suffix = current.slice(current.length - order)

    for (const source of sources) {
      const lastStart = source.length - order - 1
      for (let offset = 0; offset <= lastStart; offset++) {
        if (!sameSlice(source, offset, suffix)) continue

        const continuationStart = offset + order
        const tokens = source
          .slice(continuationStart, continuationStart + maxDraftTokens)
        if (tokens.length === 0) continue

        const key = tokens.join(',')
        const existing = candidates.get(key)
        if (!existing) {
          candidates.set(key, {
            tokens: [...tokens],
            matchOrder: order,
            occurrences: 1,
          })
        } else if (existing.matchOrder === order) {
          existing.occurrences += 1
        }
      }
    }
  }

  return [...candidates.values()]
    .sort((left, right) =>
      right.matchOrder - left.matchOrder
      || right.occurrences - left.occurrences
      || left.tokens.length - right.tokens.length
      || left.tokens.join(',').localeCompare(right.tokens.join(',')),
    )
    .slice(0, maxCandidates)
}

export class BoundedSessionTokenMemory {
  private modelKey: string | null = null
  private tokens: number[] = []

  constructor(readonly maxTokens: number) {
    if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
      throw new RangeError('maxTokens must be a positive integer')
    }
  }

  append(modelKey: string, tokens: readonly number[]): void {
    if (this.modelKey !== modelKey) {
      this.modelKey = modelKey
      this.tokens = []
    }

    if (tokens.length === 0) return
    this.tokens.push(...tokens)

    if (this.tokens.length > this.maxTokens) {
      this.tokens = this.tokens.slice(this.tokens.length - this.maxTokens)
    }
  }

  snapshot(modelKey: string): number[] {
    return this.modelKey === modelKey ? [...this.tokens] : []
  }

  /**
   * Restore caller-owned persisted memory without taking ownership of storage.
   * Consumers decide whether IndexedDB persistence is appropriate and can scope
   * or delete it with their normal privacy controls.
   */
  restore(modelKey: string, tokens: readonly number[]): void {
    this.modelKey = modelKey
    this.tokens = [...tokens.slice(-this.maxTokens)]
  }

  clear(): void {
    this.modelKey = null
    this.tokens = []
  }
}

export interface SubmissionWindowSample {
  steps: number
  elapsedMs: number
}

/**
 * Choose the smallest submission window whose measured cost per step is within
 * the requested tolerance of the fastest candidate. This mirrors the paper's
 * bias toward short windows when extra batching buys little throughput.
 */
export function chooseSubmissionWindow(
  samples: readonly SubmissionWindowSample[],
  tolerance = 0.1,
): number {
  if (samples.length === 0) {
    throw new RangeError('at least one submission-window sample is required')
  }
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError('tolerance must be a finite non-negative number')
  }

  const measured = samples.map((sample) => {
    if (!Number.isInteger(sample.steps) || sample.steps <= 0) {
      throw new RangeError('window steps must be positive integers')
    }
    if (!Number.isFinite(sample.elapsedMs) || sample.elapsedMs <= 0) {
      throw new RangeError('elapsedMs must be a finite positive number')
    }
    return {
      steps: sample.steps,
      perStepMs: sample.elapsedMs / sample.steps,
    }
  })

  const fastest = Math.min(...measured.map((sample) => sample.perStepMs))
  const threshold = fastest * (1 + tolerance)

  return measured
    .filter((sample) => sample.perStepMs <= threshold)
    .sort((left, right) => left.steps - right.steps)[0].steps
}

export interface SpeculationSample {
  width: number
  acceptedTokens: number
  elapsedMs: number
}

export interface SpeculationPlanOptions {
  targetWindowMs: number
  maxStepMs?: number
  maxWidth?: number
  maxWindowSteps?: number
  minSamples?: number
}

export interface SpeculationPlan {
  width: number
  windowSteps: number
}

interface WidthStats {
  samples: number
  acceptedTokensEma: number
  stepMsEma: number
}

export interface AdaptiveSpeculationControllerOptions {
  /**
   * Exponential smoothing factor for online measurements. Higher values react
   * faster to changing contention/clock behavior; must be in (0, 1].
   */
  smoothing?: number
}

/**
 * Empirical controller for width/window selection.
 *
 * It deliberately learns from observed accepted-token throughput rather than
 * assuming wider speculation is faster. Online costs use an exponential moving
 * average so stale calibration can yield to current device conditions. This is
 * substrate-neutral policy only; callers still need a runtime that can actually
 * verify multiple candidates.
 */
export class AdaptiveSpeculationController {
  private readonly widths: number[]
  private readonly stats = new Map<number, WidthStats>()
  private readonly smoothing: number

  constructor(
    widths: readonly number[] = [1, 2, 4, 8, 16, 32],
    options: AdaptiveSpeculationControllerOptions = {},
  ) {
    const normalized = [...new Set(widths)]
      .filter((width) => Number.isInteger(width) && width > 0)
      .sort((a, b) => a - b)

    if (!normalized.includes(1)) normalized.unshift(1)
    this.widths = normalized

    const smoothing = options.smoothing ?? 0.25
    if (!Number.isFinite(smoothing) || smoothing <= 0 || smoothing > 1) {
      throw new RangeError('smoothing must be a finite number in (0, 1]')
    }
    this.smoothing = smoothing
  }

  record(sample: SpeculationSample): void {
    if (!this.widths.includes(sample.width)) {
      throw new RangeError(`unsupported speculation width: ${sample.width}`)
    }
    if (!Number.isFinite(sample.acceptedTokens) || sample.acceptedTokens < 0) {
      throw new RangeError('acceptedTokens must be a finite non-negative number')
    }
    if (!Number.isFinite(sample.elapsedMs) || sample.elapsedMs <= 0) {
      throw new RangeError('elapsedMs must be a finite positive number')
    }

    const previous = this.stats.get(sample.width)
    if (!previous) {
      this.stats.set(sample.width, {
        samples: 1,
        acceptedTokensEma: sample.acceptedTokens,
        stepMsEma: sample.elapsedMs,
      })
      return
    }

    this.stats.set(sample.width, {
      samples: previous.samples + 1,
      acceptedTokensEma:
        previous.acceptedTokensEma
        + this.smoothing * (sample.acceptedTokens - previous.acceptedTokensEma),
      stepMsEma:
        previous.stepMsEma
        + this.smoothing * (sample.elapsedMs - previous.stepMsEma),
    })
  }

  selectPlan(options: SpeculationPlanOptions): SpeculationPlan {
    if (!Number.isFinite(options.targetWindowMs) || options.targetWindowMs <= 0) {
      throw new RangeError('targetWindowMs must be a finite positive number')
    }

    const minSamples = positiveInteger(options.minSamples, 1)
    const maxWindowSteps = positiveInteger(options.maxWindowSteps, 16)
    const maxWidth = options.maxWidth ?? Number.POSITIVE_INFINITY
    const maxStepMs = options.maxStepMs ?? Number.POSITIVE_INFINITY

    let selectedWidth = 1
    let selected: WidthStats | undefined
    let bestThroughput = Number.NEGATIVE_INFINITY

    for (const width of this.widths) {
      if (width > maxWidth) continue
      const stats = this.stats.get(width)
      if (!stats || stats.samples < minSamples || stats.stepMsEma > maxStepMs) {
        continue
      }
      const throughput = stats.acceptedTokensEma / stats.stepMsEma
      if (
        throughput > bestThroughput
        || (throughput === bestThroughput && width < selectedWidth)
      ) {
        bestThroughput = throughput
        selectedWidth = width
        selected = stats
      }
    }

    if (!selected) return { width: 1, windowSteps: 1 }

    return {
      width: selectedWidth,
      windowSteps: Math.max(
        1,
        Math.min(maxWindowSteps, Math.floor(options.targetWindowMs / selected.stepMsEma)),
      ),
    }
  }
}
