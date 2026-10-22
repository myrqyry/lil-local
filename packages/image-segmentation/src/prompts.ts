import { validatePointInImage, type Point } from './contracts'

/** SAM 2.1 uses 1 for foreground/include and 0 for background/exclude. */
export interface SamPointPrompt extends Point {
  label: 0 | 1
}

export const MAX_SAM_PROMPTS = 16

/**
 * Prepare ONE candidate object, described by up to 16 signed points.
 * The processor receives one point group and corresponding int64 point labels.
 */
export function prepareSamPointPrompts(
  prompts: readonly SamPointPrompt[],
  sourceWidth: number,
  sourceHeight: number,
  inferenceWidth: number,
  inferenceHeight: number,
): { input_points: number[][][]; input_labels: number[][] } {
  if (!Array.isArray(prompts) || prompts.length === 0 ||
      prompts.length > MAX_SAM_PROMPTS) {
    throw new Error('Provide one to 16 SAM points, including at least one include point')
  }
  if (!Number.isInteger(inferenceWidth) || !Number.isInteger(inferenceHeight) ||
      inferenceWidth < 1 || inferenceHeight < 1) {
    throw new Error('Invalid inference image size')
  }
  if (prompts.some(prompt => !prompt || (prompt.label !== 0 && prompt.label !== 1))) {
    throw new Error('Invalid SAM point label: use include 1 or exclude 0')
  }
  if (!prompts.some(prompt => prompt.label === 1)) {
    throw new Error('SAM selection requires at least one include point')
  }
  const points = prompts.map(prompt => {
    validatePointInImage(prompt, sourceWidth, sourceHeight)
    return [
      Math.min(inferenceWidth - 1, prompt.x * inferenceWidth / sourceWidth),
      Math.min(inferenceHeight - 1, prompt.y * inferenceHeight / sourceHeight),
    ]
  })
  return {
    input_points: [points],
    input_labels: [prompts.map(prompt => prompt.label)],
  }
}
