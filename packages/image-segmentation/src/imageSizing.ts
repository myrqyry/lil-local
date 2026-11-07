/** Return the resized image: Transformers.js RawImage.resize() does NOT mutate its receiver. */
export async function boundInferenceImage<T extends {
  width: number
  height: number
  resize(width: number, height: number): Promise<T>
}>(image: T): Promise<T> {
  const ratio = Math.min(1, 512 / Math.max(image.width, image.height))
  if (ratio === 1) return image
  return image.resize(
    Math.max(1, Math.round(image.width * ratio)),
    Math.max(1, Math.round(image.height * ratio)),
  )
}
