/**
 * Experimental Ternary Bonsai 2B ONNX WebGPU qualification.
 * This probe NEVER downloads a graph on import or uses a fallback EP.
 * It must pass actual browser inference before promotion to Lil Local models.
 */
export const BONSAI2_REPO = 'danyalmalik/Ternary-Bonsai-2-2B-ONNX-WebGPU'
export const BONSAI2_REVISION = 'a3c26c4d363186f744b20e08d585ce43e2ce7e8d'
export const BONSAI2_SOURCE = 'https://huggingface.co/' + BONSAI2_REPO + '/resolve/' + BONSAI2_REVISION

/**
 * Audited on 2026-10-21 from exact SHA-256 verified, immutable decoder graph.
 * Graph examined with onnx.load(..., load_external_data=False), recursively
 * visiting top-level nodes and If branches (18 Scan, 0 Loop, 18 If).
 * That file is ONLY 1.05 MB; none of the large weight blobs were downloaded.
 */
export const BONSAI2_DECODER_AUDIT = {
  status: 'failed-webgpu-audit',
  path: 'onnx/decoder_model_merged_q2f16.onnx',
  size: 1_054_038,
  sha256: 'a4b3e8a9633cfffa35104ccbf2bd748c26f36cd8b1281c8443d47b82ecd0268f',
  scanNodes: 18,
  loopNodes: 0,
  ifNodes: 18,
  logitOutputShape: '[batch, sequence, 248320] (variable batch/sequence)',
} as const

export function assessBonsaiBrowserPath(
  features: Pick<ReadonlySet<string>, 'has'>,
  info: { vendor?: string; architecture?: string },
) {
  return {
    state: 'blocked-unsafe-decoder' as const,
    hardware: assessBonsaiGpu(features, info),
    reason: 'Pinned Bonsai 2B ONNX decoder has 18 Scan nodes. ONNX Runtime WebGPU cannot run Scan and uses CPU/GPU token round-trips; a shader-f16 adapter alone cannot make it a qualified browser text model. Use the PrismML native GGUF runtime or a separately audited rewritten graph.',
  }
}

export type BonsaiGpuEligibility = {
  state: 'unsupported' | 'eligible-for-qualification'
  vendor: string
  architecture: string
  shaderF16: boolean
  reason: string
}

/**
 * This exported ONNX graph requires shader-f16 for its quantized embedding
 * operator. Eligibility is *not* model readiness or inference qualification.
 */
export function assessBonsaiGpu(
  features: Pick<ReadonlySet<string>, 'has'>,
  info: { vendor?: string; architecture?: string },
): BonsaiGpuEligibility {
  const shaderF16 = features.has('shader-f16')
  return {
    state: shaderF16 ? 'eligible-for-qualification' : 'unsupported',
    vendor: info.vendor || 'unknown',
    architecture: info.architecture || 'unknown',
    shaderF16,
    reason: shaderF16
      ? 'shader-f16 is available. ONNX decoder execution, GPU placement, tokenizer and text generation are still unverified.'
      : 'The model embedding uses GatherBlockQuantized with FP16 output and requires the WebGPU shader-f16 feature. This adapter does not expose it.',
  }
}

/** No network/model downloads and no GPU allocation; caller explicitly initiates this check. */
export async function inspectBonsaiGpu(): Promise<BonsaiGpuEligibility> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return assessBonsaiGpu(new Set(), {})
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) return assessBonsaiGpu(new Set(), {})
  return assessBonsaiGpu(adapter.features, adapter.info)
}

export interface BonsaiEmbeddingProbe {
  adapter: { vendor: string; architecture: string }
  dims: readonly number[]
  finiteSamples: number
  loadMs: number
  inferMs: number
}

/** Prove the 4-bit quantized GatherBlockQuantized embedding stage on GPU-requested ORT. */
export async function probeBonsaiEmbedding(
  graphUrl: string,
  weightUrl: string,
): Promise<BonsaiEmbeddingProbe> {
  if (!navigator.gpu) throw new Error('WebGPU required; no WASM fallback')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('WebGPU adapter unavailable')
  const eligibility = assessBonsaiGpu(adapter.features, adapter.info)
  if (!eligibility.shaderF16) {
    throw new Error('Bonsai ONNX WebGPU preflight refused before model downloads: ' + eligibility.reason)
  }
  const info = adapter.info
  const ort = await import('onnxruntime-web/webgpu')
  let session: import('onnxruntime-web').InferenceSession | null = null
  const began = performance.now()
  try {
    session = await ort.InferenceSession.create(graphUrl, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      externalData: [{ path: 'embed_tokens_q2f16.onnx_data', data: weightUrl }],
    })
    if (session.inputNames.length !== 1 || session.inputNames[0] !== 'input_ids' ||
        session.outputNames.length !== 1 || session.outputNames[0] !== 'inputs_embeds') {
      throw new Error('Embedding graph ABI mismatch')
    }
    const loadMs = Math.round(performance.now() - began)
    const start = performance.now()
    const outputs = await session.run({
      input_ids: new ort.Tensor('int64', BigInt64Array.from([1n, 42n, 100n]), [1, 3]),
    })
    const tensor = outputs.inputs_embeds
    if (!tensor || tensor.dims.join(',') !== '1,3,2048') {
      throw new Error('Unexpected embedding dimensions')
    }
    const data = tensor.data
    const finiteSamples = Array.from(data.slice(0, 24), Number).filter(Number.isFinite).length
    if (!finiteSamples) throw new Error('All sampled embeddings non-finite')
    return {
      adapter: { vendor: info.vendor, architecture: info.architecture },
      dims: tensor.dims,
      finiteSamples,
      loadMs,
      inferMs: Math.round(performance.now() - start),
    }
  } finally {
    await session?.release()
  }
}
