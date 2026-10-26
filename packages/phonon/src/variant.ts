import type { Backend } from '@lil-local/inference-core';
import type { PhononVariant } from './manifest';

/** Match the checkpoint's compute types to the *actual* browser GPU features. */
export function choosePhononVariant(
  backend: Backend, shaderF16: boolean, requested?: PhononVariant,
): PhononVariant {
  if (backend !== 'webgpu' && backend !== 'wasm') throw new Error('Phonon supports only WebGPU and WASM');
  const variant = requested ?? (backend === 'webgpu' ? (shaderF16 ? 'q4f16' : 'q4') : 'fp32');
  if (variant === 'q4f16' && backend === 'wasm') throw new Error('q4f16 requires WebGPU');
  if (variant === 'q4f16' && !shaderF16) throw new Error('q4f16 requires WebGPU shader-f16');
  return variant;
}
