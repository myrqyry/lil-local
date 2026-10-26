import type { BrowserSubstrate } from '@lil-local/inference-core'
import { SharpOnnxWebGpu, type SharpOnnxOptions, type SharpRunInput, type SharpRunOutput } from './runtime'

/** Integrates Apple's single-image Gaussian predictor into Lil Local's
 * existing engine-neutral lifecycle without introducing a second scheduler.
 * The returned descriptor is inert until BrowserSubstrateRunner.load().
 */
export function createSharpSubstrate(options: SharpOnnxOptions): BrowserSubstrate<SharpRunInput, SharpRunOutput> {
  return {
    id: 'apple-sharp-onnx',
    format: 'onnx',
    capabilities: ['image-to-3d'],
    requestedBackends: ['webgpu'],
    async create(_onProgress, signal) {
      const runner = new SharpOnnxWebGpu(options)
      await runner.load(signal)
      return {
        infer: (image) => runner.run(image),
        dispose: () => runner.dispose(),
      }
    },
  }
}
