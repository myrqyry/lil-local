import { getGlobalLiteRt } from '@litertjs/core'
import { createHttpAssetResolver } from '@lil-local/inference-core'
import { createLiteRtRuntime } from '@lil-local/runtime-litert'
import { assertAddOutput } from './add-parity'

/** Opt-in real Chromium/WASM qualification using installed package tarballs. */
export async function runPackedTinyLitert(): Promise<{
  resolvedBackend: string; maxError: number; cached: boolean; events: string[]; cases: number
}> {
  const path = '/_tiny/model.tflite'
  const context = await createLiteRtRuntime({ backend: 'wasm', assets: createHttpAssetResolver(location.origin) })
  try {
    const compiled = await context.liteRt.loadModel(path)
    const cached = (await context.liteRt.loadModel(path)) === compiled
    const inputs = compiled.getInputDetails()
    if (inputs.length !== 2) throw new Error(`Expected 2 inputs, got ${inputs.length}`)
    const wasm = getGlobalLiteRt().liteRtWasm as unknown as Record<string, unknown>
    if (typeof wasm.LiteRtTensorBuffer !== 'function') throw new Error('Missing WASM LiteRtTensorBuffer')
    let maxError = 0
    const cases = [[3, 5], [-2, 7]] as const
    for (const [left, right] of cases) {
      const tensors = inputs.map((input, index) =>
        context.liteRt.createTensor(
          new Float32Array(input.shape.reduce((total, size) => total * size, 1)).fill(index === 0 ? left : right),
          [...input.shape],
        ),
      )
      try {
        const output = await context.liteRt.predict(path, tensors)
        if (!Array.isArray(output) || output.length !== 1) throw new Error('Expected one output tensor')
        try {
          const values = context.liteRt.readTensor<Float32Array>(output[0])
          maxError = Math.max(maxError, assertAddOutput(values, left, right))
        } finally {
          for (const tensor of output) tensor.delete()
        }
      } finally {
        for (const tensor of tensors) tensor.delete()
      }
    }
    const info = context.liteRt.getModelInfo(path)
    if (!info || info.resolvedBackend !== 'wasm' || info.fallbackCount !== 0) {
      throw new Error(`Unexpected backend receipt ${JSON.stringify(info)}`)
    }
    const events = context.liteRt.getTelemetry().map(record => record.event)
    if (events.filter(event => event === 'compile').length !== 1 ||
        events.filter(event => event === 'inference').length !== cases.length) {
      throw new Error(`Unexpected telemetry: ${events.join(',')}`)
    }
    return { resolvedBackend: info.resolvedBackend, maxError, cached, events, cases: cases.length }
  } finally {
    context.liteRt.dispose()
  }
}
