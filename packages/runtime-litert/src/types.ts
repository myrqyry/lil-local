import type { Backend, InferenceDiagnostics, RuntimeContext } from '@litert-playground/inference-core'
import type { CompiledModel, Tensor, TensorDetails } from '@litertjs/core'
import type { BackendPreference } from './capabilities'
import type { InferenceCoordinator } from './coordinator'

export interface WebNNRuntimeOptions {
  devicePreference?: 'cpu' | 'gpu' | 'npu'
  powerPreference?: 'default' | 'high-performance' | 'low-power'
  precision?: 'fp32' | 'fp16'
}

export interface LiteRtModelOptions {
  accelerator?: BackendPreference
  supportedBackends?: Partial<Record<Backend, boolean | 'experimental'>>
  webNNOptions?: WebNNRuntimeOptions
  signal?: AbortSignal
  onProgress?: (progress: { loadedBytes: number; totalBytes?: number }) => void
  /** Optional inference/preflight queue priority; does not affect model identity or loading. */
  inferencePriority?: number
}

export type LiteRtTypedArray = Float32Array | Int32Array | Int8Array | Uint8Array
export type LiteRtModelInput = Tensor | Tensor[] | Record<string, Tensor>
export type LiteRtModelOutput = Tensor[] | Record<string, Tensor>

export interface LiteRtModelInfo {
  modelPath: string
  requestedBackend: BackendPreference
  resolvedBackend: Backend
  compileDurationMs: number
  fallbackCount: number
}

export interface LiteRtTelemetryRecord extends LiteRtModelInfo {
  event: 'compile' | 'inference' | 'preflight'
  timestamp: string
  inferenceDurationMs?: number
  inputCount?: number
  outputCount?: number
  tensorCopyCount: number
}

export interface LiteRtPreflightOptions extends LiteRtModelOptions {
  signature?: string
  maxTensorElements?: number
  createInputs?: (inputDetails: readonly TensorDetails[]) => LiteRtModelInput
}

export interface LiteRtPreflightResult extends LiteRtModelInfo {
  inputDetails: readonly TensorDetails[]
  outputDetails: readonly TensorDetails[]
  outputCount: number
  inferenceDurationMs: number
}

export interface LiteRtRuntimeOptions {
  packageName?: string
  assetBase?: string
  backend?: BackendPreference
  assets: RuntimeContext['assets']
  signal?: AbortSignal
  supportedBackends?: Partial<Record<Backend, boolean | 'experimental'>>
  webNNOptions?: WebNNRuntimeOptions
  coordinator?: InferenceCoordinator
  telemetryLimit?: number
  onTelemetry?: (record: LiteRtTelemetryRecord) => void
}

export interface ManagedLiteRtRuntime {
  loadModel(path: string, options?: LiteRtModelOptions): Promise<CompiledModel>
  loadNpy(path: string, signal?: AbortSignal): Promise<Float32Array>
  fetchBuffer(path: string, signal?: AbortSignal): Promise<ArrayBuffer>
  predict(
    path: string,
    input: LiteRtModelInput,
    options?: LiteRtModelOptions & { label?: string },
  ): Promise<LiteRtModelOutput>
  predictWithSignature(
    path: string,
    signature: string,
    input: LiteRtModelInput,
    options?: LiteRtModelOptions & { label?: string },
  ): Promise<LiteRtModelOutput>
  preflight(path: string, options?: LiteRtPreflightOptions): Promise<LiteRtPreflightResult>
  getModelInfo(path: string, options?: LiteRtModelOptions): LiteRtModelInfo | undefined
  getDiagnostics(path: string, options?: LiteRtModelOptions): InferenceDiagnostics | undefined
  getTelemetry(): readonly LiteRtTelemetryRecord[]
  clearTelemetry(): void
  createTensor(data: LiteRtTypedArray, shape: number[]): Tensor
  readTensor<T extends LiteRtTypedArray>(tensor: Tensor): T
  supportsGpuBufferTensors(): boolean
  disposeModel(path: string): void
  dispose(): void
}

export interface ManagedLiteRtRuntimeContext extends RuntimeContext {
  liteRt: ManagedLiteRtRuntime
}
