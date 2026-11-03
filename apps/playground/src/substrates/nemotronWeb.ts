/** Browser-only Nemotron ASR contract. We deliberately do not re-use Android GGUF/JNI. */
export const MODEL_REVISION = '8364d9e2dd9da23789b480bdbba9e423717e42ee'
export const MODEL_ROOT = `https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/${MODEL_REVISION}`
export type NemotronBackend = 'auto' | 'webgpu' | 'wasm'

export function chooseNemotronBackend(webGpuAvailable: boolean, request: NemotronBackend): 'webgpu' | 'wasm' {
  if (request === 'wasm') return 'wasm'
  if (webGpuAvailable) return 'webgpu'
  if (request === 'webgpu') throw new Error('WebGPU is unavailable; explicit WebGPU request cannot fall back')
  return 'wasm'
}

export function validateAudioSamples(samples: Float32Array): void {
  if (!samples.length) throw new Error('Audio is empty')
  if (!samples.every(Number.isFinite)) throw new Error('Audio must contain finite samples')
}
