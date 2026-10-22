import { validateRegion, type SegmentationRegion } from './contracts'

/** Convert model returned [1, candidate_count, H, W] binary masks to scored, editable regions. */
export function decodeSamMaskCandidates(
  values: Uint8Array,
  dims: readonly number[],
  scores: Float32Array,
): SegmentationRegion[] {
  const [batch, count, height, width] = dims
  if (dims.length !== 4 || batch !== 1 || !Number.isInteger(count) ||
      !Number.isInteger(height) || !Number.isInteger(width) ||
      count < 1 || count > 3 || height < 1 || width < 1 ||
      width > 1024 || height > 1024 || width * height > 1024 * 1024) {
    throw new Error('Unexpected SAM mask output shape')
  }
  if (values.length !== count * width * height || scores.length !== count) {
    throw new Error('Unexpected SAM mask or score length')
  }
  if (Array.from(scores).some(score => !Number.isFinite(score))) {
    throw new Error('SAM returned invalid quality scores')
  }
  const candidates = Array.from(scores, (score, index) => ({ score, index }))
    .filter(entry => entry.score >= 0.05)
    .sort((a, b) => b.score - a.score)
  if (!candidates.length) throw new Error('SAM returned no confident masks for that click; try another point')
  const regions: SegmentationRegion[] = []
  for (const { score, index } of candidates) {
    const binary = values.subarray(index * width * height, (index + 1) * width * height)
    const data = Uint8Array.from(binary, v => v > 0 ? 255 : 0)
    if (!data.some(v => v > 0)) continue
    const region = { label: 'sam-object-' + (regions.length + 1), width, height, data, score }
    validateRegion(region)
    regions.push(region)
  }
  if (!regions.length) throw new Error('SAM generated no visible region for that click')
  return regions
}
