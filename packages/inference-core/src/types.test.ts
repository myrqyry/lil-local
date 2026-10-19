import { describe, expect, it } from 'vitest'
import {
  checkAudioValid,
  checkMultiVectorEmbeddingValid,
  InferenceError,
  type AudioInferenceResult,
  type Capability,
  type InferenceDiagnostics,
  type InferenceReceipt,
  type ModelAsset,
  type ModelVerification,
  type InferenceResult,
  type Pipeline,
  type RetrievalResult,
  type VisionLanguageResult,
} from './index'

describe('inference core', () => {
  it('keeps audio validation model-agnostic', () => {
    expect(checkAudioValid(new Float32Array(), 24_000, 1, 0)).toContain('audio: empty samples')
  })

  it('exposes shared pipeline and error contracts', () => {
    const pipeline: Pipeline<unknown, AudioInferenceResult> | null = null
    expect(pipeline).toBeNull()
    expect(new InferenceError('CANCELLED', 'cancelled').code).toBe('CANCELLED')
  })

  it('covers the LFM model families in the capability registry', () => {
    const lfm: Capability[] = [
      'image-text-to-text',
      'token-classification',
      'text-classification',
      'reranking',
      'multi-vector-embedding',
      'policy-classification',
    ]
    expect(lfm).toHaveLength(6)
  })

  it('narrows multi-vector embeddings and retrieval results', () => {
    const multi: InferenceResult = {
      kind: 'multi-vector-embedding',
      values: new Float32Array([1, 0, 0, 1]),
      tokens: 2,
      dimensions: 2,
    }
    expect(multi).toMatchObject({ kind: 'multi-vector-embedding', tokens: 2, dimensions: 2 })
    expect(checkMultiVectorEmbeddingValid(multi.values, multi.tokens, multi.dimensions)).toEqual([])
    expect(checkMultiVectorEmbeddingValid(new Float32Array(0), 1, 2)).toContain(
      'multi-vector-embedding: empty values',
    )

    const ranked: RetrievalResult = { id: 'episode-1', score: 0.91, payload: { episode: 'x' } }
    expect(ranked.score).toBe(0.91)
  })

  it('carries a reasoning channel on text results and composes vision-language contracts', () => {
    const text: InferenceResult = { kind: 'text', text: 'answer', reasoning: 'thinking' }
    expect(text.reasoning).toBe('thinking')

    const vl: VisionLanguageResult = {
      text: 'found it',
      boxes: [{ x: 0, y: 0, width: 10, height: 10, score: 0.9 }],
      receipt: {
        modelId: 'lfm2.5-vl-450m',
        pipelineVersion: '0.1.0',
        backend: 'webgpu',
        timestamp: new Date(0).toISOString(),
        loadMs: 1,
        compileMs: 1,
        inferenceMs: 1,
        inputSummary: 'image 10x10 + prompt',
        outputSummary: 'text + 1 box',
        warnings: [],
      },
    }
    expect(vl.boxes).toHaveLength(1)
    expect(vl.receipt.modelId).toBe('lfm2.5-vl-450m')
  })

  it('keeps manifest qualification metadata serializable', () => {
    const asset: ModelAsset = {
      id: 'model',
      path: 'model.tflite',
      mimeType: 'application/octet-stream',
      role: 'model',
    }
    const verification: ModelVerification = {
      assets: 'pass',
      compile: 'pass',
      inference: 'pass',
      output: 'pass',
      qualification: 'qualified',
      upstreamRevision: 'revision-1',
      environments: [{ browser: 'chromium', backend: 'webgpu', runtime: 'litert.js' }],
      expectedOutput: {
        outputShape: [1, 600],
        labels: { assetId: 'labels', count: 600, mapping: 'index-to-label' },
        behavior: ['softmax scores'],
      },
      lastVerifiedAt: new Date(0).toISOString(),
    }

    expect(JSON.parse(JSON.stringify({ asset, verification }))).toEqual({ asset, verification })
  })

  it('keeps diagnostics and receipt values serializable', () => {
    const diagnostics: InferenceDiagnostics = {
      packageName: '@lil-local/test',
      modelId: 'model',
      requestedBackend: 'auto',
      resolvedBackend: 'wasm',
      cacheHit: true,
      compileMs: 4,
      inferenceMs: 5,
      fallbackCount: 1,
      queueMs: 2,
      error: { code: 'INFERENCE_FAILED', message: 'failed', stage: 'run', asset: 'model' },
    }
    const receipt: Pick<InferenceReceipt, 'modelId' | 'diagnostics'> = {
      modelId: 'model',
      diagnostics,
    }

    expect(JSON.parse(JSON.stringify({ diagnostics, receipt }))).toEqual({ diagnostics, receipt })
  })
})
