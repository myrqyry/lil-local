import { beforeEach, describe, expect, it, vi } from 'vitest'
import { loadAndCompile, loadLiteRt, setWebGpuDevice, Tensor } from '@litertjs/core'
import { createLiteRtRuntime } from './context'
import { InferenceCoordinator } from './coordinator'

vi.mock('@litertjs/core', async () => {
  const actual = await vi.importActual<typeof import('@litertjs/core')>('@litertjs/core')
  return {
    ...actual,
    loadLiteRt: vi.fn(),
    loadAndCompile: vi.fn(),
    setWebGpuDevice: vi.fn(),
  }
})

describe('managed LiteRT runtime', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('navigator', {})
    vi.mocked(loadLiteRt).mockResolvedValue({} as never)
    vi.mocked(loadAndCompile).mockResolvedValue({} as never)
  })

  it('caches compiled models and emits compile telemetry', async () => {
    const model = {}
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) }
    const onTelemetry = vi.fn()
    const context = await createLiteRtRuntime({ backend: 'wasm', assets, onTelemetry })

    await context.liteRt.loadModel('vision.tflite')
    await context.liteRt.loadModel('vision.tflite')

    expect(assets.resolve).toHaveBeenCalledTimes(1)
    expect(loadAndCompile).toHaveBeenCalledTimes(1)
    expect(context.liteRt.getModelInfo('vision.tflite')).toMatchObject({
      modelPath: 'vision.tflite',
      requestedBackend: 'wasm',
      resolvedBackend: 'wasm',
      fallbackCount: 0,
    })
    expect(context.liteRt.getTelemetry()).toHaveLength(1)
    expect(onTelemetry).toHaveBeenCalledWith(expect.objectContaining({ event: 'compile' }))
  })

  it('deduplicates concurrent model loads', async () => {
    let releaseCompile!: (model: unknown) => void
    const compileGate = new Promise((resolve) => { releaseCompile = resolve })
    vi.mocked(loadAndCompile).mockReturnValue(compileGate as never)
    const assets = { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) }
    const context = await createLiteRtRuntime({ backend: 'wasm', assets })

    const first = context.liteRt.loadModel('shared.tflite')
    const second = context.liteRt.loadModel('shared.tflite')
    await vi.waitFor(() => expect(loadAndCompile).toHaveBeenCalledTimes(1))
    releaseCompile({})

    await expect(Promise.all([first, second])).resolves.toHaveLength(2)
    expect(assets.resolve).toHaveBeenCalledTimes(1)
    expect(loadAndCompile).toHaveBeenCalledTimes(1)
  })

  it('keeps a shared load alive when one deduplicated caller aborts', async () => {
    let releaseCompile!: (model: unknown) => void
    const compileGate = new Promise((resolve) => { releaseCompile = resolve })
    vi.mocked(loadAndCompile).mockReturnValue(compileGate as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })
    const firstController = new AbortController()
    const secondController = new AbortController()

    const first = context.liteRt.loadModel('shared.tflite', { signal: firstController.signal })
    const second = context.liteRt.loadModel('shared.tflite', { signal: secondController.signal })
    await vi.waitFor(() => expect(loadAndCompile).toHaveBeenCalledTimes(1))

    firstController.abort()
    await expect(first).rejects.toMatchObject({ code: 'CANCELLED' })

    const model = {}
    releaseCompile(model)
    await expect(second).resolves.toBe(model)
    expect(loadAndCompile).toHaveBeenCalledTimes(1)
  })

  it('does not let a disposed pending load repopulate the cache', async () => {
    let releaseFirst!: (model: unknown) => void
    let releaseSecond!: (model: unknown) => void
    const firstGate = new Promise((resolve) => { releaseFirst = resolve })
    const secondGate = new Promise((resolve) => { releaseSecond = resolve })
    vi.mocked(loadAndCompile)
      .mockReturnValueOnce(firstGate as never)
      .mockReturnValueOnce(secondGate as never)

    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    const first = context.liteRt.loadModel('replaceable.tflite')
    await vi.waitFor(() => expect(loadAndCompile).toHaveBeenCalledTimes(1))

    context.liteRt.disposeModel('replaceable.tflite')
    await expect(first).rejects.toMatchObject({ code: 'CANCELLED' })

    const second = context.liteRt.loadModel('replaceable.tflite')
    await vi.waitFor(() => expect(loadAndCompile).toHaveBeenCalledTimes(2))

    const staleModel = { delete: vi.fn() }
    releaseFirst(staleModel)
    await vi.waitFor(() => expect(staleModel.delete).toHaveBeenCalledTimes(1))
    expect(context.liteRt.getModelInfo('replaceable.tflite')).toBeUndefined()

    const freshModel = {}
    releaseSecond(freshModel)
    await expect(second).resolves.toBe(freshModel)
    expect(context.liteRt.getModelInfo('replaceable.tflite')).toMatchObject({
      modelPath: 'replaceable.tflite',
    })
  })

  it('reloads a fresh model after disposing the cached model', async () => {
    const firstModel = {}
    const secondModel = {}
    vi.mocked(loadAndCompile)
      .mockResolvedValueOnce(firstModel as never)
      .mockResolvedValueOnce(secondModel as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    await expect(context.liteRt.loadModel('reloadable.tflite')).resolves.toBe(firstModel)
    context.liteRt.disposeModel('reloadable.tflite')
    await expect(context.liteRt.loadModel('reloadable.tflite')).resolves.toBe(secondModel)

    expect(loadAndCompile).toHaveBeenCalledTimes(2)
  })

  it('keeps WebGPU and WASM model cache entries separate', async () => {
    const device = {}
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue(device) }) },
    })
    const webGpuModel = {}
    const wasmModel = {}
    vi.mocked(loadAndCompile)
      .mockResolvedValueOnce(webGpuModel as never)
      .mockResolvedValueOnce(wasmModel as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    await expect(context.liteRt.loadModel('alternating.tflite', { accelerator: 'webgpu' }))
      .resolves.toBe(webGpuModel)
    await expect(context.liteRt.loadModel('alternating.tflite', { accelerator: 'wasm' }))
      .resolves.toBe(wasmModel)

    expect(loadAndCompile).toHaveBeenCalledTimes(2)
    expect(loadAndCompile).toHaveBeenNthCalledWith(
      1,
      expect.any(Uint8Array),
      expect.objectContaining({ accelerator: 'webgpu' }),
    )
    expect(loadAndCompile).toHaveBeenNthCalledWith(
      2,
      expect.any(Uint8Array),
      expect.objectContaining({ accelerator: 'wasm' }),
    )
  })

  it('does not execute inference queued before disposal', async () => {
    let releaseFirst!: (output: unknown) => void
    const firstGate = new Promise((resolve) => { releaseFirst = resolve })
    const firstOutput = [{}]
    const secondOutput = [{}]
    const model = {
      run: vi.fn()
        .mockReturnValueOnce(firstGate)
        .mockResolvedValueOnce(secondOutput),
    }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      coordinator: new InferenceCoordinator(),
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    await context.liteRt.loadModel('queued-disposal.tflite')
    const first = context.liteRt.predict('queued-disposal.tflite', {} as never)
    await vi.waitFor(() => expect(model.run).toHaveBeenCalledTimes(1))
    const second = context.liteRt.predict('queued-disposal.tflite', {} as never)
    context.liteRt.dispose()
    releaseFirst(firstOutput)

    await expect(first).resolves.toEqual(firstOutput)
    await expect(second).rejects.toMatchObject({ code: 'INFERENCE_FAILED' })
    expect(model.run).toHaveBeenCalledTimes(1)
  })

  it('honors predict inferencePriority through the actual LiteRT coordinator', async () => {
    const order: string[] = []
    let release!: () => void
    const gate = new Promise<void>((resolve) => { release = resolve })
    const model = {
      run: vi.fn(async (input: { id: string }) => {
        order.push(input.id)
        if (input.id === 'active') await gate
        return [{}]
      }),
    }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const coordinator = new InferenceCoordinator()
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      coordinator,
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })
    await context.liteRt.loadModel('shared.tflite')

    const first = context.liteRt.predict('shared.tflite', { id: 'active' } as never)
    await vi.waitFor(() => expect(coordinator.snapshot().active).toBe(true))
    const ordinary = context.liteRt.predict('shared.tflite', { id: 'ordinary' } as never)
    const urgent = context.liteRt.predict(
      'shared.tflite',
      { id: 'urgent' } as never,
      { inferencePriority: 8 },
    )
    await vi.waitFor(() => expect(coordinator.snapshot().queued).toBe(2))

    release()
    await Promise.all([first, ordinary, urgent])
    expect(order).toEqual(['active', 'urgent', 'ordinary'])
    expect(loadAndCompile).toHaveBeenCalledTimes(1)
    expect(context.liteRt.getModelInfo('shared.tflite')).toBeDefined()
  })

  it('returns JSON-serializable runtime diagnostics', async () => {
    const model = { run: vi.fn().mockResolvedValue([{}]) }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      packageName: '@example/consumer',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    await context.liteRt.predict('diagnostics.tflite', {} as never)

    const diagnostics = context.liteRt.getDiagnostics('diagnostics.tflite')
    expect(diagnostics).toMatchObject({
      packageName: '@example/consumer',
      modelId: 'diagnostics.tflite',
      requestedBackend: 'wasm',
      resolvedBackend: 'wasm',
      cacheHit: true,
      fallbackCount: 0,
    })
    expect(diagnostics?.compileMs).toEqual(expect.any(Number))
    expect(diagnostics?.inferenceMs).toEqual(expect.any(Number))
    expect(diagnostics?.queueMs).toEqual(expect.any(Number))
    expect(JSON.parse(JSON.stringify(diagnostics))).toEqual(diagnostics)
  })

  it('uses WebNN before WASM when WebGPU compilation fails in auto mode', async () => {
    const device = {}
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: vi.fn().mockResolvedValue(device) }) },
      ml: {},
    })
    vi.mocked(loadAndCompile)
      .mockRejectedValueOnce(new Error('webgpu compile failed'))
      .mockResolvedValueOnce({} as never)
    const context = await createLiteRtRuntime({
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
      webNNOptions: { devicePreference: 'gpu', precision: 'fp16' },
    })

    await context.liteRt.loadModel('model.tflite')

    expect(setWebGpuDevice).toHaveBeenCalledWith(device)
    expect(loadAndCompile).toHaveBeenNthCalledWith(
      2,
      expect.any(Uint8Array),
      expect.objectContaining({
        accelerator: 'webnn',
        webNNOptions: { devicePreference: 'gpu', precision: 'fp16' },
      }),
    )
    expect(context.backend).toBe('webnn')
    expect(context.liteRt.getModelInfo('model.tflite')?.fallbackCount).toBe(1)
  })

  it('preflights a model through the shared coordinator and records a receipt-like trace', async () => {
    const output = {} as never
    const model = {
      getInputDetails: vi.fn().mockReturnValue([{ shape: [1, 4], dtype: 'float32' }]),
      getOutputDetails: vi.fn().mockReturnValue([{ shape: [1, 2], dtype: 'float32' }]),
      run: vi.fn().mockResolvedValue([output]),
    }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    const result = await context.liteRt.preflight('model.tflite', { createInputs: () => [] })

    expect(model.run).toHaveBeenCalledWith([])
    expect(result).toMatchObject({ resolvedBackend: 'wasm', outputCount: 1, fallbackCount: 0 })
    expect(context.liteRt.getTelemetry().map((entry) => entry.event)).toEqual(['compile', 'preflight'])
  })

  it('releases internally generated preflight inputs and discarded outputs', async () => {
    const input = { delete: vi.fn() } as unknown as Tensor
    const output = { delete: vi.fn() } as unknown as Tensor
    const tensorSpy = vi.spyOn(Tensor, 'fromTypedArray').mockReturnValue(input)
    const model = {
      getInputDetails: vi.fn().mockReturnValue([{ shape: [1, 4], dtype: 'float32' }]),
      getOutputDetails: vi.fn().mockReturnValue([{ shape: [1, 2], dtype: 'float32' }]),
      run: vi.fn().mockResolvedValue([output]),
    }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    try {
      await context.liteRt.preflight('cleanup.tflite')
      expect(input.delete).toHaveBeenCalledTimes(1)
      expect(output.delete).toHaveBeenCalledTimes(1)
    } finally {
      tensorSpy.mockRestore()
    }
  })

  it('releases internally generated preflight inputs when inference fails', async () => {
    const input = { delete: vi.fn() } as unknown as Tensor
    const tensorSpy = vi.spyOn(Tensor, 'fromTypedArray').mockReturnValue(input)
    const model = {
      getInputDetails: vi.fn().mockReturnValue([{ shape: [1, 4], dtype: 'float32' }]),
      getOutputDetails: vi.fn().mockReturnValue([]),
      run: vi.fn().mockRejectedValue(new Error('preflight failed')),
    }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    try {
      await expect(context.liteRt.preflight('cleanup-failure.tflite')).rejects.toMatchObject({
        code: 'INFERENCE_FAILED',
      })
      expect(input.delete).toHaveBeenCalledTimes(1)
    } finally {
      tensorSpy.mockRestore()
    }
  })

  it('supports named-signature prediction and inference telemetry', async () => {
    const result = [{}] as never
    const model = { run: vi.fn().mockResolvedValue(result) }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })
    const input = {} as never

    await context.liteRt.predictWithSignature('model.tflite', 'decode', input, { label: 'decode-step' })

    expect(model.run).toHaveBeenCalledWith('decode', input)
    const telemetry = context.liteRt.getTelemetry()
    expect(telemetry[telemetry.length - 1]).toMatchObject({
      event: 'inference',
      modelPath: 'model.tflite',
      resolvedBackend: 'wasm',
      outputCount: 1,
    })
  })

  it('does not let telemetry callback failures change runtime results', async () => {
    const model = { run: vi.fn().mockResolvedValue([{}]) }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const onTelemetry = vi.fn(() => {
      throw new Error('observer failed')
    })
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
      onTelemetry,
    })

    await expect(context.liteRt.loadModel('telemetry.tflite')).resolves.toBe(model)
    await expect(context.liteRt.predict('telemetry.tflite', {} as never)).resolves.toHaveLength(1)

    expect(loadAndCompile).toHaveBeenCalledTimes(1)
    expect(onTelemetry).toHaveBeenCalledTimes(2)
    expect(context.liteRt.getTelemetry().map((entry) => entry.event)).toEqual(['compile', 'inference'])
  })

  it('bounds telemetry history for long-running consumers', async () => {
    const model = { run: vi.fn().mockResolvedValue([{}]) }
    vi.mocked(loadAndCompile).mockResolvedValue(model as never)
    const context = await createLiteRtRuntime({
      backend: 'wasm',
      telemetryLimit: 2,
      assets: { resolve: vi.fn().mockResolvedValue(new ArrayBuffer(8)) },
    })

    await context.liteRt.predict('model.tflite', {} as never)
    await context.liteRt.predict('model.tflite', {} as never)
    await context.liteRt.predict('model.tflite', {} as never)

    expect(context.liteRt.getTelemetry()).toHaveLength(2)
    expect(context.liteRt.getTelemetry().every((entry) => entry.event === 'inference')).toBe(true)
  })
})
