import test from 'node:test'
import assert from 'node:assert/strict'
import { median, selectMeasuredWorkgroup } from './webgpu-matmul-selection.mjs'

test('median is order-independent and rejects invalid samples', () => {
  assert.equal(median([3, 1, 2]), 2)
  assert.equal(median([4, 1, 3, 2]), 2.5)
  assert.throws(() => median([]), /samples/)
  assert.throws(() => median([1, NaN]), /samples/)
})

test('does not select a workgroup when timing differences are noise', () => {
  const result = selectMeasuredWorkgroup([
    { workgroup: 8, correct: true, samplesMs: [0.262, 0.275, 0.287] },
    { workgroup: 16, correct: true, samplesMs: [0.263, 0.275, 0.275] },
  ])
  assert.equal(result.selected, null)
  assert.equal(result.reason, 'indistinguishable')
})

test('requires independently correct results and a meaningful timing margin', () => {
  assert.deepEqual(selectMeasuredWorkgroup([
    { workgroup: 8, correct: true, samplesMs: [1.1, 1.2, 1.3] },
    { workgroup: 16, correct: true, samplesMs: [0.5, 0.6, 0.7] },
  ]), { selected: 16, reason: 'measured-advantage' })
  assert.equal(selectMeasuredWorkgroup([
    { workgroup: 8, correct: false, samplesMs: [0.1, 0.1, 0.1] },
    { workgroup: 16, correct: true, samplesMs: [0.4, 0.5, 0.6] },
  ]).selected, null)
  assert.throws(() => selectMeasuredWorkgroup([]), /candidates/)
})
