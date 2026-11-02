import { getGlobalLiteRt } from '@litertjs/core'
import { createHttpAssetResolver } from '@lil-local/inference-core'
import { createLiteRtRuntime } from '@lil-local/runtime-litert'

/** Explicit opt-in browser qualification from installed package tarballs. */
export async function runPackedTinyLitert(): Promise<{
  resolvedBackend: string
  maxError: number
  cached: boolean
  events: string[]
}> {
  const path = '/_tiny/model.tflite'
  const context = await createLiteRtRuntime({
    backend: 'wasm',
    assets: createHttpAssetResolver(location.origin),
  })
  try {
    const compiled = await context.liteRt.loadModel(path)
    const cached = await context.liteRt.loadModel(path) === compiled
    const inputs = compiled.getInputDetails()
    if (inputs.length !== 2) throw new Error(`Expected 2 inputs, got ${inputs.length}`)
    const wasm = getGlobalLiteRt().liteRtWasm as unknown as Record<string, unknown>
    if (typeof wasm.LiteRtTensorBuffer !== 'function') {
      throw new Error(`Loaded WASM is missing LiteRtTensorBuffer (typeof=${typeof wasm.LiteRtTensorBuffer}; available tensor keys: ${Object.keys(wasm).filter(key => key.toLowerCase().includes('tensor')).slice(0, 20).join(',')})`)
    }
    const tensors = inputs.map((input, index) =>
      context.liteRt.createTensor(
        new Float32Array(input.shape.reduce((total, size) => total * size, 1)).fill(index === 0 ? 3 : 5),
        [...input.shape],
      ),
    )
    try {
      const output = await context.liteRt.predict(path, tensors)
      if (!Array.isArray(output) || output.length !== 1) throw new Error('Expected one output tensor')
      try {
        const values = context.liteRt.readTensor<Float32Array>(output[0])
        if (!values.length) throw new Error('Empty output')
        let maxError = 0
        for (const value of values) {
          if (!Number.isFinite(value)) throw new Error('Non-finite model output')
          maxError = Math.max(maxError, Math.abs(value - 8))
        }
        if (maxError > 1e-5) throw new Error(`Numerical mismatch: expected 3+5=8; max error=${maxError}`)
        const info = context.liteRt.getModelInfo(path)
        if (!info || info.resolvedBackend !== 'wasm' || info.fallbackCount !== 0) {
          throw new Error(`Unexpected backend receipt ${JSON.stringify(info)}`)
        }
        const events = context.liteRt.getTelemetry().map(event => event.event)
        if (events.filter(event => event === 'compile').length !== 1 || !events.includes('inference')) {
          throw new Error(`Unexpected telemetry: ${events.join(',')}`)
        }
        return { resolvedBackend: info.resolvedBackend, maxError, cached, events }
      } finally {
        output[0].delete()
      }
    } finally {
      for (const tensor of tensors) tensor.delete()
    }
  } finally {
    context.liteRt.dispose()
  }
}
