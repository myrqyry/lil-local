import type { RuntimeContext, ModelManifest, Pipeline, PipelineStatus } from '@lil-local/inference-core'
import type { ManagedLiteRtRuntime, LiteRtModelOptions } from '@lil-local/runtime-litert'
import { prepareSaliencyInput, decodeSaliencyMask, type RgbaSource, type ForegroundMask } from './mask'

export const u2NetSaliencyManifest: ModelManifest = {
  modelId:'u2net-salient-320',
  name:'U²-Net saliency — LiteRT 320px candidate',
  version:'0.1.0',
  capabilities:['image-segmentation'],
  backends:{wasm:'experimental'},
  // Download is source-reported exact size. Resident memory is a conservative
  // preflight budget, NOT an observed runtime peak.
  memory:{downloadBytes:88_230_272,residentBytes:1_000_000_000},
  assets:[{id:'model',
    path:'https://huggingface.co/litert-community/U-2-Net/resolve/defc203955a46f3cc760deb8c76d96f10331a46c/u2net_fp16.tflite',
    bytes:88_230_272,
    sha256:'dd338f190a538ca3de9792b20b7617038cd56f8e440f38ff25da5554f32b9df2',
    role:'foreground saliency model'}],
  verification:{assets:'untested',compile:'untested',inference:'untested',
    output:'untested',qualification:'limited', upstreamRevision:'defc203955a46f3cc760deb8c76d96f10331a46c',
    expectedOutput:{preprocessing:['320x320 RGB float32 NCHW','per-image max scaling then ImageNet mean/std'],
      outputShape:[1,1,320,320],behavior:['saliency probability, NOT semantic part labels',
        'WASM default for first controlled qualification; inference/output not yet qualified']}},
}

/**
 * Opt-in model session. Does not load/download until load() is invoked.
 * Default WASM is a conservative first-qualification backend, not a GPU ban.
 */
export class SaliencySegmentationPipeline implements Pipeline<RgbaSource, ForegroundMask> {
  readonly manifest = u2NetSaliencyManifest
  status: PipelineStatus = 'idle'
  private runtime: ManagedLiteRtRuntime | null = null
  private readonly modelOptions: LiteRtModelOptions = { accelerator:'wasm', supportedBackends:{wasm:true} }
  private get modelUrl(): string { return this.manifest.assets[0].path }

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return
    if (this.status === 'running') throw new Error('Cannot reload during inference')
    this.status = 'loading'
    const runtime = context.liteRt as ManagedLiteRtRuntime
    try {
      const compiled = await runtime.loadModel(this.modelUrl, this.modelOptions)
      const input = compiled.getInputDetails()
      const output = compiled.getOutputDetails()
      const matches = (shape: readonly number[], expected: number[]) =>
        shape.length === expected.length && shape.every((dim,i)=>dim===expected[i])
      if (input.length!==1 || output.length!==1 ||
        input[0].dtype !== 'float32' || output[0].dtype !== 'float32' ||
        !matches(Array.from(input[0].shape),[1,3,320,320]) || !matches(Array.from(output[0].shape),[1,1,320,320])) {
        throw new Error('U²-Net saliency model signature mismatch; refusing unverified tensor binding')
      }
      this.runtime = runtime
      this.status = 'ready'
    } catch (cause) {
      runtime.disposeModel(this.modelUrl)
      this.status = 'error'
      throw cause
    }
  }

  async run(source: RgbaSource, _config?: unknown, signal?: AbortSignal): Promise<ForegroundMask> {
    if (this.status!=='ready' || !this.runtime) throw new Error('Foreground model not ready')
    if (signal?.aborted) throw new Error('Foreground inference cancelled')
    const runtime = this.runtime
    // Preprocessing happens before status changes, so invalid images cannot wedge the session.
    const values = prepareSaliencyInput(source)
    this.status = 'running'
    let input: ReturnType<ManagedLiteRtRuntime['createTensor']> | undefined
    try {
      input = runtime.createTensor(values,[1,3,320,320])
      const outputs = await runtime.predict(this.modelUrl,{input},{
        ...this.modelOptions, signal, label:'foreground-segmentation',
      })
      const tensors = Array.isArray(outputs) ? outputs : Object.values(outputs)
      try {
        if (tensors.length !== 1) throw new Error('Expected one foreground mask output')
        const values = runtime.readTensor<Float32Array>(tensors[0])
        const frame = decodeSaliencyMask(values,320,320,source.width,source.height)
        frame.modelId = this.manifest.modelId
        frame.backend = runtime.getModelInfo(this.modelUrl,this.modelOptions)?.resolvedBackend ?? 'wasm'
        return frame
      } finally {
        tensors.forEach(t=>t.delete())
      }
    } finally {
      input?.delete()
      this.status = 'ready'
    }
  }

  async dispose(): Promise<void> {
    this.runtime?.disposeModel(this.modelUrl)
    this.runtime = null
    this.status = 'disposed'
  }
}
