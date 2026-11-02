/** Independent oracle for Google's elementwise add_10x10 TFLite fixture. */
export function assertAddOutput(
  actual: Float32Array,
  left: number,
  right: number,
  tolerance = 1e-5,
): number {
  if (!actual.length) throw new Error('Empty model output')
  if (![left, right, tolerance].every(Number.isFinite) || tolerance < 0) {
    throw new Error('Invalid numerical oracle inputs')
  }
  const expected = left + right
  let maxError = 0
  for (const value of actual) {
    if (!Number.isFinite(value)) throw new Error('Non-finite model output')
    maxError = Math.max(maxError, Math.abs(value - expected))
  }
  if (maxError > tolerance) {
    throw new Error(`Numerical mismatch: expected ${left}+${right}=${expected}; max error=${maxError}`)
  }
  return maxError
}
