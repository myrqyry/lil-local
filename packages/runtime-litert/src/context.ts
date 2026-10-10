import {
  loadAndCompile,
  loadLiteRt,
  setWebGpuDevice,
  Tensor,
  type CompiledModel,
  type TensorDetails,
} from '@litertjs/core'
import {
  InferenceError,
  type InferenceDiagnostics,
  type Backend,
  type RuntimeCapabilities,
} from '@litert-playground/inference-core'
import { rankBackends, probeRuntimeCapabilities, selectBackend } from './capabilities'
import { inferenceCoordinator as defaultCoordinator } from './coordinator'
import { parseNpy } from './npy'
import type {
  LiteRtModelInfo,
  LiteRtModelInput,
  LiteRtModelOptions,
  LiteRtModelOutput,
  LiteRtPreflightOptions,
  LiteRtPreflightResult,
  LiteRtRuntimeOptions,
  LiteRtTelemetryRecord,
  LiteRtTypedArray,
  ManagedLiteRtRuntime,
  ManagedLiteRtRuntimeContext,
  WebNNRuntimeOptions,
} from './types'

interface LoadedModel extends LiteRtModelInfo {
  model: CompiledModel
}

interface PendingLoad {
  key: string
  path: string
  generation: number
  controller: AbortController
  promise: Promise<LoadedModel>
  subscribers: number
  settled: boolean
  detachRuntimeAbort?: () => void
}

const DEFAULT_TELEMETRY_LIMIT = 512
// ponytail: pin @litertjs/core CDN fallback here; bump when peer dep bumps (workspace:*)
const LITERT_CORE_CDN = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/'

function runtimeBase(assetBase?: string): string {
  const pageBase = (globalThis as { location?: { href: string } }).location?.href ?? 'http://localhost/'
  return new URL(assetBase ?? LITERT_CORE_CDN, pageBase).href
}

function stableOptions(value: unknown): string {
  return JSON.stringify(value ?? {})
}

function countInput(input: LiteRtModelInput): number {
  if (Array.isArray(input)) return input.length
  if (input instanceof Tensor) return 1
  return Object.keys(input).length
}

function countOutput(output: LiteRtModelOutput): number {
  return Array.isArray(output) ? output.length : Object.keys(output).length
}

function isAbort(cause: unknown, signal?: AbortSignal): boolean {
  return signal?.aborted === true || (cause instanceof Error && cause.name === 'AbortError')
}

function tensorFrom(data: LiteRtTypedArray, shape: number[]): Tensor {
  return Tensor.fromTypedArray(data as Parameters<typeof Tensor.fromTypedArray>[0], shape)
}

function zeroTensor(details: TensorDetails, maxElements: number): Tensor {
  const shape = [...details.shape]
  if (shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new InferenceError('INVALID_INPUT', `Dynamic or invalid preflight tensor shape: ${shape.join('x')}`, {
      stage: 'preflight',
    })
  }

  const size = shape.reduce((total, dimension) => total * dimension, 1)
  if (size > maxElements) {
    throw new InferenceError('INVALID_INPUT', `Preflight tensor is too large: ${size} elements`, {
      stage: 'preflight',
    })
  }

  // LiteRT.js currently types TensorDetails.dtype as float32 | int32 | uint8.
  // Keep the string switch permissive so known quantized adapters can still
  // provide int8 metadata through createInputs or future runtime versions.
  switch (details.dtype as string) {
    case 'float32':
      return tensorFrom(new Float32Array(size), shape)
    case 'int32':
      return tensorFrom(new Int32Array(size), shape)
    case 'int8':
      return tensorFrom(new Int8Array(size), shape)
    case 'uint8':
      return tensorFrom(new Uint8Array(size), shape)
    default:
      throw new InferenceError('INVALID_INPUT', `Unsupported preflight tensor type: ${details.dtype}`, {
        stage: 'preflight',
      })
  }
}

function safeDeleteTensor(tensor: Tensor): void {
  try {
    tensor.delete()
  } catch {
    // Cleanup must never turn an otherwise successful operation into a failure.
  }
}

function disposeTensors(value: LiteRtModelInput | LiteRtModelOutput): void {
  const tensors = Array.isArray(value)
    ? value
    : value instanceof Tensor
      ? [value]
      : Object.values(value)

  for (const tensor of new Set(tensors)) safeDeleteTensor(tensor)
}

function safeDeleteModel(model: CompiledModel): void {
  try {
    ;(model as CompiledModel & { delete?: () => void }).delete?.()
  } catch {
    // Some LiteRT.js model wrappers do not expose an explicit delete hook.
  }
}

class LiteRtRuntimeManager implements ManagedLiteRtRuntime {
  private readonly models = new Map<string, LoadedModel>()
  private readonly pendingLoads = new Map<string, PendingLoad>()
  private readonly modelGenerations = new Map<string, number>()
  private readonly telemetry: LiteRtTelemetryRecord[] = []
  private readonly diagnostics = new Map<string, InferenceDiagnostics>()
  private resolvedBackend: Backend
  private webGpuDevice: Promise<unknown> | null = null
  private tensorCopies = 0
  private disposed = false

  constructor(
    private readonly options: LiteRtRuntimeOptions,
    private readonly capabilities: RuntimeCapabilities,
    defaultBackend: Backend,
  ) {
    this.resolvedBackend = defaultBackend
  }

  get backend(): Backend {
    return this.resolvedBackend
  }

  async loadModel(path: string, options: LiteRtModelOptions = {}): Promise<CompiledModel> {
    this.assertUsable()
    const key = this.modelKey(path, options)
    const cached = this.models.get(key)
    if (cached) return cached.model

    const signal = options.signal ?? this.options.signal
    const pending = this.pendingLoads.get(key) ?? this.startPendingLoad(path, options, key)
    return this.awaitPendingLoad(pending, signal)
  }

  async loadNpy(path: string, signal?: AbortSignal): Promise<Float32Array> {
    this.assertUsable()
    return parseNpy(await this.resolve(path, signal))
  }

  async fetchBuffer(path: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    this.assertUsable()
    return this.resolve(path, signal)
  }

  async predict(
    path: string,
    input: LiteRtModelInput,
    options: LiteRtModelOptions & { label?: string } = {},
  ): Promise<LiteRtModelOutput> {
    return this.infer(path, input, undefined, options)
  }

  async predictWithSignature(
    path: string,
    signature: string,
    input: LiteRtModelInput,
    options: LiteRtModelOptions & { label?: string } = {},
  ): Promise<LiteRtModelOutput> {
    return this.infer(path, input, signature, options)
  }

  async preflight(path: string, options: LiteRtPreflightOptions = {}): Promise<LiteRtPreflightResult> {
    this.assertUsable()
    const signal = options.signal ?? this.options.signal
    const model = await this.loadModel(path, options)
    const info = this.requireInfo(path, options)
    const inputDetails = model.getInputDetails()
    const outputDetails = model.getOutputDetails()
    const ownsInputs = options.createInputs === undefined
    const inputs = options.createInputs?.(inputDetails)
      ?? inputDetails.map((details) => zeroTensor(details, options.maxTensorElements ?? 10_000_000))

    let output: LiteRtModelOutput | undefined
    let startedAt = 0
    try {
      output = await (this.options.coordinator ?? defaultCoordinator).run(async () => {
        startedAt = performance.now()
        return (options.signature
          ? await model.run(options.signature, inputs)
          : await model.run(inputs)) as LiteRtModelOutput
      }, signal, `litert-preflight:${path}`, { priority: options.inferencePriority })

      const inferenceDurationMs = performance.now() - startedAt
      const result: LiteRtPreflightResult = {
        ...info,
        inputDetails,
        outputDetails,
        outputCount: countOutput(output),
        inferenceDurationMs,
      }
      this.record({
        ...info,
        event: 'preflight',
        timestamp: new Date().toISOString(),
        inferenceDurationMs,
        inputCount: countInput(inputs),
        outputCount: result.outputCount,
        tensorCopyCount: this.tensorCopies,
      })
      return result
    } catch (cause) {
      if (cause instanceof InferenceError) throw cause
      if (isAbort(cause, signal)) {
        throw new InferenceError('CANCELLED', `Preflight cancelled for ${path}`, { stage: 'preflight', cause })
      }
      throw new InferenceError('INFERENCE_FAILED', `Preflight failed for ${path}`, { stage: 'preflight', cause })
    } finally {
      if (ownsInputs) disposeTensors(inputs)
      if (output) disposeTensors(output)
    }
  }

  getModelInfo(path: string, options: LiteRtModelOptions = {}): LiteRtModelInfo | undefined {
    const entry = this.models.get(this.modelKey(path, options))
      ?? [...this.models.values()].find((candidate) => candidate.modelPath === path)
    if (!entry) return undefined
    const { model: _model, ...info } = entry
    return info
  }

  getDiagnostics(path: string, options: LiteRtModelOptions = {}): InferenceDiagnostics | undefined {
    return this.diagnostics.get(this.modelKey(path, options))
      ?? [...this.diagnostics.values()].find((candidate) => candidate.modelId === path)
  }

  getTelemetry(): readonly LiteRtTelemetryRecord[] {
    return [...this.telemetry]
  }

  clearTelemetry(): void {
    this.telemetry.length = 0
    this.diagnostics.clear()
    this.tensorCopies = 0
  }

  createTensor(data: LiteRtTypedArray, shape: number[]): Tensor {
    this.assertUsable()
    return tensorFrom(data, shape)
  }

  readTensor<T extends LiteRtTypedArray>(tensor: Tensor): T {
    this.assertUsable()
    this.tensorCopies += 1
    return tensor.toTypedArray() as T
  }

  supportsGpuBufferTensors(): boolean {
    return typeof (Tensor as unknown as { fromGpuBuffer?: unknown }).fromGpuBuffer === 'function'
  }

  disposeModel(path: string): void {
    this.bumpGeneration(path)

    for (const [key, entry] of this.models) {
      if (entry.modelPath !== path) continue
      safeDeleteModel(entry.model)
      this.models.delete(key)
    }

    for (const [key, pending] of this.pendingLoads) {
      if (pending.path !== path) continue
      this.pendingLoads.delete(key)
      pending.controller.abort()
    }
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true

    for (const entry of this.models.values()) safeDeleteModel(entry.model)
    for (const pending of this.pendingLoads.values()) pending.controller.abort()

    this.models.clear()
    this.pendingLoads.clear()
    this.modelGenerations.clear()
    this.telemetry.length = 0
    this.webGpuDevice = null
    this.tensorCopies = 0
  }

  private startPendingLoad(path: string, options: LiteRtModelOptions, key: string): PendingLoad {
    const controller = new AbortController()
    const runtimeSignal = this.options.signal
    let detachRuntimeAbort: (() => void) | undefined

    if (runtimeSignal) {
      if (runtimeSignal.aborted) {
        controller.abort()
      } else {
        const onAbort = () => controller.abort()
        runtimeSignal.addEventListener('abort', onAbort, { once: true })
        detachRuntimeAbort = () => runtimeSignal.removeEventListener('abort', onAbort)
      }
    }

    const generation = this.currentGeneration(path)
    let pending!: PendingLoad
    const promise = this.compile(path, { ...options, signal: controller.signal }, key, generation)
      .finally(() => {
        pending.settled = true
        pending.detachRuntimeAbort?.()
        if (this.pendingLoads.get(key) === pending) this.pendingLoads.delete(key)
      })

    pending = {
      key,
      path,
      generation,
      controller,
      promise,
      subscribers: 0,
      settled: false,
      detachRuntimeAbort,
    }
    this.pendingLoads.set(key, pending)
    return pending
  }

  private async awaitPendingLoad(pending: PendingLoad, signal?: AbortSignal): Promise<CompiledModel> {
    if (signal?.aborted || pending.controller.signal.aborted) {
      throw this.cancelledLoad(pending.path)
    }

    pending.subscribers += 1
    try {
      const entry = await new Promise<LoadedModel>((resolve, reject) => {
        let finished = false
        const signals = [...new Set(
          [signal, pending.controller.signal].filter((candidate): candidate is AbortSignal => candidate !== undefined),
        )]

        const cleanup = () => {
          for (const candidate of signals) candidate.removeEventListener('abort', onAbort)
        }
        const finish = (callback: () => void) => {
          if (finished) return
          finished = true
          cleanup()
          callback()
        }
        const onAbort = () => finish(() => reject(this.cancelledLoad(pending.path)))

        for (const candidate of signals) candidate.addEventListener('abort', onAbort, { once: true })
        pending.promise.then(
          (value) => finish(() => resolve(value)),
          (cause) => finish(() => reject(cause)),
        )

        if (signals.some((candidate) => candidate.aborted)) onAbort()
      })
      return entry.model
    } finally {
      pending.subscribers -= 1
      if (
        pending.subscribers === 0
        && !pending.settled
        && this.pendingLoads.get(pending.key) === pending
      ) {
        this.pendingLoads.delete(pending.key)
        pending.controller.abort()
      }
    }
  }

  private async compile(
    path: string,
    options: LiteRtModelOptions,
    key: string,
    generation: number,
  ): Promise<LoadedModel> {
    const signal = options.signal
    const onProgress = options.onProgress
    const requestedBackend = options.accelerator ?? this.options.backend ?? 'auto'
    const supported = options.supportedBackends ?? this.options.supportedBackends ?? {}
    const candidates = rankBackends(this.capabilities, supported, requestedBackend)
    if (candidates.length === 0) {
      throw new InferenceError('BACKEND_UNAVAILABLE', `No usable backend for ${path} (${requestedBackend})`, {
        stage: 'compile',
        asset: path,
      })
    }

    const bytes = await this.resolve(path, signal, onProgress)
    let lastError: unknown

    for (let index = 0; index < candidates.length; index += 1) {
      const backend = candidates[index]
      if (signal?.aborted || !this.isGenerationCurrent(path, generation)) {
        throw this.cancelledLoad(path)
      }

      const compileStartedAt = performance.now()
      try {
        await this.prepareBackend(backend)
        const model = await loadAndCompile(
          new Uint8Array(bytes),
          this.compileOptions(backend, options.webNNOptions ?? this.options.webNNOptions),
        )

        if (this.disposed) {
          safeDeleteModel(model)
          this.assertUsable()
        }
        if (signal?.aborted || !this.isGenerationCurrent(path, generation)) {
          safeDeleteModel(model)
          throw this.cancelledLoad(path)
        }

        const entry: LoadedModel = {
          model,
          modelPath: path,
          requestedBackend,
          resolvedBackend: backend,
          compileDurationMs: performance.now() - compileStartedAt,
          fallbackCount: index,
        }
        this.models.set(key, entry)
        this.resolvedBackend = backend
        const { model: _model, ...info } = entry
        this.diagnostics.set(key, {
          packageName: this.options.packageName ?? '@litert-playground/runtime-litert',
          modelId: path,
          requestedBackend,
          resolvedBackend: backend,
          cacheHit: false,
          compileMs: info.compileDurationMs,
          fallbackCount: info.fallbackCount,
        })
        this.record({
          ...info,
          event: 'compile',
          timestamp: new Date().toISOString(),
          tensorCopyCount: this.tensorCopies,
        })
        return entry
      } catch (cause) {
        lastError = cause
        if (backend === 'webgpu') this.webGpuDevice = null
        if (cause instanceof InferenceError && cause.message.includes('disposed')) throw cause
        if (cause instanceof InferenceError && cause.code === 'CANCELLED') throw cause
        if (isAbort(cause, signal)) {
          throw new InferenceError('CANCELLED', `Model load cancelled for ${path}`, {
            stage: 'compile',
            asset: path,
            cause,
          })
        }
        if (requestedBackend !== 'auto') break
      }
    }

    throw new InferenceError('MODEL_COMPILE_FAILED', `Failed to compile ${path} on ${candidates.join(', ')}`, {
      stage: 'compile',
      asset: path,
      cause: lastError,
    })
  }

  private async infer(
    path: string,
    input: LiteRtModelInput,
    signature: string | undefined,
    options: LiteRtModelOptions & { label?: string },
  ): Promise<LiteRtModelOutput> {
    this.assertUsable()
    const signal = options.signal ?? this.options.signal
    const model = await this.loadModel(path, options)
    const info = this.requireInfo(path, options)
    const key = this.modelKey(path, options)
    const queuedAt = performance.now()
    let startedAt = 0

    try {
      const output = await (this.options.coordinator ?? defaultCoordinator).run(async () => {
        this.assertUsable()
        startedAt = performance.now()
        return (signature ? await model.run(signature, input) : await model.run(input)) as LiteRtModelOutput
      }, signal, options.label ?? `litert:${path}`, { priority: options.inferencePriority })

      const inferenceDurationMs = performance.now() - startedAt
      this.diagnostics.set(key, {
        packageName: this.options.packageName ?? '@litert-playground/runtime-litert',
        modelId: path,
        requestedBackend: info.requestedBackend,
        resolvedBackend: info.resolvedBackend,
        cacheHit: true,
        compileMs: info.compileDurationMs,
        inferenceMs: inferenceDurationMs,
        fallbackCount: info.fallbackCount,
        queueMs: startedAt - queuedAt,
      })
      this.record({
        ...info,
        event: 'inference',
        timestamp: new Date().toISOString(),
        inferenceDurationMs,
        inputCount: countInput(input),
        outputCount: countOutput(output),
        tensorCopyCount: this.tensorCopies,
      })
      return output
    } catch (cause) {
      const error = cause instanceof InferenceError
        ? cause
        : new InferenceError(
            isAbort(cause, signal) ? 'CANCELLED' : 'INFERENCE_FAILED',
            isAbort(cause, signal) ? `Inference cancelled for ${path}` : `Inference failed for ${path}`,
            { stage: 'inference', cause },
          )
      this.diagnostics.set(key, {
        packageName: this.options.packageName ?? '@litert-playground/runtime-litert',
        modelId: path,
        requestedBackend: info.requestedBackend,
        resolvedBackend: info.resolvedBackend,
        cacheHit: true,
        compileMs: info.compileDurationMs,
        fallbackCount: info.fallbackCount,
        queueMs: startedAt > 0 ? startedAt - queuedAt : undefined,
        error: {
          code: error.code,
          message: error.message,
          stage: error.stage,
          asset: error.asset,
        },
      })
      if (cause instanceof InferenceError) throw cause
      if (isAbort(cause, signal)) {
        throw new InferenceError('CANCELLED', `Inference cancelled for ${path}`, { stage: 'inference', cause })
      }
      throw new InferenceError('INFERENCE_FAILED', `Inference failed for ${path}`, { stage: 'inference', cause })
    }
  }

  private requireInfo(path: string, options: LiteRtModelOptions): LiteRtModelInfo {
    const info = this.getModelInfo(path, options)
    if (!info) throw new InferenceError('INFERENCE_FAILED', `Model ${path} is not loaded`)
    return info
  }

  private modelKey(path: string, options: LiteRtModelOptions): string {
    return [
      path,
      options.accelerator ?? this.options.backend ?? 'auto',
      stableOptions(options.supportedBackends ?? this.options.supportedBackends),
      stableOptions(options.webNNOptions ?? this.options.webNNOptions),
    ].join('::')
  }

  private currentGeneration(path: string): number {
    return this.modelGenerations.get(path) ?? 0
  }

  private isGenerationCurrent(path: string, generation: number): boolean {
    return this.currentGeneration(path) === generation
  }

  private bumpGeneration(path: string): void {
    this.modelGenerations.set(path, this.currentGeneration(path) + 1)
  }

  private cancelledLoad(path: string): InferenceError {
    return new InferenceError('CANCELLED', `Model load cancelled for ${path}`, {
      stage: 'compile',
      asset: path,
    })
  }

  private async resolve(path: string, signal?: AbortSignal, onProgress?: (progress: { loadedBytes: number; totalBytes?: number }) => void): Promise<ArrayBuffer> {
    const effectiveSignal = signal ?? this.options.signal
    try {
      return await this.options.assets.resolve({ id: path, path }, { signal: effectiveSignal, onProgress })
    } catch (cause) {
      if (isAbort(cause, effectiveSignal)) {
        throw new InferenceError('CANCELLED', `Asset fetch cancelled for ${path}`, { asset: path, cause })
      }
      throw new InferenceError('ASSET_FETCH_FAILED', `Failed to resolve ${path}`, { asset: path, cause })
    }
  }

  private async prepareBackend(backend: Backend): Promise<void> {
    if (backend !== 'webgpu') return
    if (!this.webGpuDevice) {
      this.webGpuDevice = (async () => {
        const gpu = (globalThis as {
          navigator?: { gpu?: { requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null> } }
        }).navigator?.gpu
        const adapter = await gpu?.requestAdapter()
        const device = await adapter?.requestDevice()
        if (!device) throw new InferenceError('BACKEND_UNAVAILABLE', 'WebGPU adapter is no longer usable')
        setWebGpuDevice(device as Parameters<typeof setWebGpuDevice>[0])
        return device
      })().catch((error) => {
        this.webGpuDevice = null
        throw error
      })
    }
    await this.webGpuDevice
  }

  private compileOptions(backend: Backend, webNNOptions?: WebNNRuntimeOptions): Parameters<typeof loadAndCompile>[1] {
    const options: { accelerator: Backend; webNNOptions?: WebNNRuntimeOptions } = { accelerator: backend }
    if (backend === 'webnn' && webNNOptions) options.webNNOptions = webNNOptions
    return options as Parameters<typeof loadAndCompile>[1]
  }

  private record(record: LiteRtTelemetryRecord): void {
    this.telemetry.push(record)
    const limit = Math.max(1, this.options.telemetryLimit ?? DEFAULT_TELEMETRY_LIMIT)
    if (this.telemetry.length > limit) this.telemetry.splice(0, this.telemetry.length - limit)

    try {
      this.options.onTelemetry?.(record)
    } catch {
      // Telemetry is observational and must never change runtime success/failure.
    }
  }

  private assertUsable(): void {
    if (this.disposed) throw new InferenceError('INFERENCE_FAILED', 'LiteRT runtime has been disposed')
  }
}

export async function createLiteRtRuntime(options: LiteRtRuntimeOptions): Promise<ManagedLiteRtRuntimeContext> {
  try {
    await loadLiteRt(new URL('wasm/', runtimeBase(options.assetBase)).href, { jspi: true })
  } catch (cause) {
    throw new InferenceError('BACKEND_UNAVAILABLE', `Failed to load LiteRT runtime: ${String(cause)}`, { cause })
  }

  const capabilities = await probeRuntimeCapabilities()
  let backend: Backend
  try {
    backend = selectBackend(capabilities, options.supportedBackends, options.backend ?? 'auto')
  } catch (cause) {
    throw new InferenceError('BACKEND_UNAVAILABLE', String(cause), { cause })
  }

  const runtime = new LiteRtRuntimeManager(options, capabilities, backend)
  return {
    get backend() {
      return runtime.backend
    },
    assets: options.assets,
    signal: options.signal,
    liteRt: runtime,
  }
}

export type { RuntimeContext } from '@litert-playground/inference-core'
