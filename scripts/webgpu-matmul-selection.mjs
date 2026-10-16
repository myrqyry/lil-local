/**
 * Experimental WebGPU microkernel measurement only. These results never
 * change LiteRT or ONNX Runtime execution. Fail closed on inaccurate output.
 */
export function median(samples) {
  if (!Array.isArray(samples) || samples.length === 0 ||
      samples.some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error('Timing samples must be nonempty, finite and nonnegative')
  }
  const sorted = [...samples].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] :
    (sorted[middle - 1] + sorted[middle]) / 2
}

/**
 * Report a candidate only with a >=10% median advantage AND nonoverlapping
 * timing samples; otherwise say inconclusive instead of inventing a winner.
 */
export function selectMeasuredWorkgroup(candidates) {
  if (!Array.isArray(candidates) || candidates.length < 2) {
    throw new Error('At least two workgroup candidates are required')
  }
  const valid = candidates.every(candidate =>
    candidate.correct === true && Number.isInteger(candidate.workgroup) &&
    candidate.workgroup > 0 && Array.isArray(candidate.samplesMs) &&
    candidate.samplesMs.length >= 3)
  if (!valid) return { selected: null, reason: 'invalid-candidate' }
  const ordered = candidates.map(candidate => ({
    ...candidate,
    central: median(candidate.samplesMs),
    slowest: Math.max(...candidate.samplesMs),
    fastest: Math.min(...candidate.samplesMs),
  })).sort((a, b) => a.central - b.central)
  const best = ordered[0]
  const next = ordered[1]
  if (best.central <= 0 || next.central <= 0 ||
      best.central / next.central > 0.9 || best.slowest >= next.fastest) {
    return { selected: null, reason: 'indistinguishable' }
  }
  return { selected: best.workgroup, reason: 'measured-advantage' }
}
