import { describe, expect, it } from 'vitest'

import {
  AdaptiveSpeculationController,
  BoundedSessionTokenMemory,
  chooseSubmissionWindow,
  collectSuffixDraftCandidates,
} from './speculation'

describe('collectSuffixDraftCandidates', () => {
  it('prefers the longest matching suffix and returns only following tokens', () => {
    const candidates = collectSuffixDraftCandidates(
      [1, 2, 3, 4],
      [[9, 2, 3, 4, 5, 6, 7]],
      { maxOrder: 3, maxDraftTokens: 2 },
    )

    expect(candidates[0]).toEqual({
      tokens: [5, 6],
      matchOrder: 3,
      occurrences: 1,
    })
  })

  it('merges duplicate continuations across page, conversation and session sources', () => {
    const candidates = collectSuffixDraftCandidates(
      [7, 8],
      [
        [0, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [4, 7, 8, 6],
      ],
      { maxOrder: 2, maxDraftTokens: 2 },
    )

    expect(candidates[0]).toEqual({
      tokens: [9, 10],
      matchOrder: 2,
      occurrences: 2,
    })
    expect(candidates[1]).toEqual({
      tokens: [6],
      matchOrder: 2,
      occurrences: 1,
    })
  })

  it('caps candidate and continuation sizes and drops matches with no continuation', () => {
    const candidates = collectSuffixDraftCandidates(
      [1],
      [[1, 2, 3, 4], [1, 5, 6, 7], [9, 1]],
      { maxOrder: 8, maxDraftTokens: 2, maxCandidates: 1 },
    )

    expect(candidates).toEqual([
      { tokens: [2, 3], matchOrder: 1, occurrences: 1 },
    ])
  })
})

describe('BoundedSessionTokenMemory', () => {
  it('keeps only the newest tokens within its explicit budget', () => {
    const memory = new BoundedSessionTokenMemory(5)
    memory.append('model-a', [1, 2, 3])
    memory.append('model-a', [4, 5, 6])

    expect(memory.snapshot('model-a')).toEqual([2, 3, 4, 5, 6])
  })

  it('clears old-session tokens when the model identity changes', () => {
    const memory = new BoundedSessionTokenMemory(8)
    memory.append('model-a', [1, 2, 3])
    memory.append('model-b', [7, 8])

    expect(memory.snapshot('model-a')).toEqual([])
    expect(memory.snapshot('model-b')).toEqual([7, 8])
  })

  it('restores caller-persisted tokens under the same explicit budget', () => {
    const memory = new BoundedSessionTokenMemory(3)
    memory.restore('model-a', [1, 2, 3, 4, 5])

    expect(memory.snapshot('model-a')).toEqual([3, 4, 5])
  })
})

describe('chooseSubmissionWindow', () => {
  it('keeps the smallest window within ten percent of the fastest measured cost per step', () => {
    expect(chooseSubmissionWindow([
      { steps: 1, elapsedMs: 10 },
      { steps: 2, elapsedMs: 18 },
      { steps: 4, elapsedMs: 34 },
    ])).toBe(2)
  })

  it('keeps one step when its cost is already close enough to the fastest', () => {
    expect(chooseSubmissionWindow([
      { steps: 1, elapsedMs: 9.2 },
      { steps: 2, elapsedMs: 18 },
      { steps: 4, elapsedMs: 34 },
    ])).toBe(1)
  })
})

describe('AdaptiveSpeculationController', () => {
  it('chooses the calibrated width with the best accepted tokens per millisecond', () => {
    const controller = new AdaptiveSpeculationController([1, 2, 4])
    controller.record({ width: 1, acceptedTokens: 1, elapsedMs: 4 })
    controller.record({ width: 2, acceptedTokens: 1.8, elapsedMs: 4 })
    controller.record({ width: 4, acceptedTokens: 2.4, elapsedMs: 8 })

    expect(controller.selectPlan({ targetWindowMs: 12 })).toEqual({
      width: 2,
      windowSteps: 3,
    })
  })

  it('rejects a faster-throughput width when its step cost exceeds the caller budget', () => {
    const controller = new AdaptiveSpeculationController([1, 2, 4])
    controller.record({ width: 1, acceptedTokens: 1, elapsedMs: 4 })
    controller.record({ width: 2, acceptedTokens: 2, elapsedMs: 5 })
    controller.record({ width: 4, acceptedTokens: 4, elapsedMs: 9 })

    expect(controller.selectPlan({ targetWindowMs: 16, maxStepMs: 6 })).toEqual({
      width: 2,
      windowSteps: 3,
    })
  })

  it('uses exponential smoothing so recent device slowdown can replace stale calibration', () => {
    const controller = new AdaptiveSpeculationController(
      [1, 2],
      { smoothing: 0.5 },
    )
    controller.record({ width: 1, acceptedTokens: 1, elapsedMs: 4 })
    for (let index = 0; index < 4; index++) {
      controller.record({ width: 2, acceptedTokens: 2, elapsedMs: 4 })
    }
    expect(controller.selectPlan({ targetWindowMs: 12 }).width).toBe(2)

    controller.record({ width: 2, acceptedTokens: 2, elapsedMs: 20 })

    expect(controller.selectPlan({ targetWindowMs: 12 }).width).toBe(1)
  })

  it('fails closed to single-token single-step decoding before calibration', () => {
    const controller = new AdaptiveSpeculationController([1, 2, 4, 8])

    expect(controller.selectPlan({ targetWindowMs: 16 })).toEqual({
      width: 1,
      windowSteps: 1,
    })
  })
})
