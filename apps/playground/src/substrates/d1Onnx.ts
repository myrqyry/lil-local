import { D1OnnxWebGpu, type D1Decision, type D1LoadPhase, type D1Question } from '@lil-local/decision-onnx'
import type { BrowserSubstrate } from '@lil-local/inference-core'

export interface D1SubstrateInput {
  state: string
  question: D1Question
}

/** Real ONNX Runtime Web implementation of the shared browser-substrate contract. */
export const d1OnnxSubstrate: BrowserSubstrate<D1SubstrateInput, D1Decision, D1LoadPhase> = {
  id: 'onnxruntime-web:d1-omni-600m',
  format: 'onnx',
  capabilities: ['reasoning'],
  requestedBackends: ['webgpu'],
  async create(onProgress, signal) {
    const model = await D1OnnxWebGpu.load(onProgress, signal)
    return {
      infer: ({ state, question }) => model.decide(state, question),
      dispose: () => model.dispose(),
    }
  },
}
