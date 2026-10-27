import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { pipeline } from '@huggingface/transformers'
import {
  createModerationSubstrate,
  MODERATION_CATEGORIES,
  MODERATION_MODEL_ID,
  MODERATION_MODEL_REVISION,
} from './index'

vi.mock('@huggingface/transformers', () => ({ pipeline: vi.fn() }))

type Classifier = ReturnType<typeof makeClassifier>

function fullScores(overrides: Partial<Record<string, number>> = {}) {
  return [
    { label: 'unsafe', score: overrides.unsafe ?? 0.2 },
    ...MODERATION_CATEGORIES.map(category => ({
      label: category,
      score: overrides[category] ?? 0.05,
    })),
  ]
}

function makeClassifier(labels: Array<{ label: string; score: number }>) {
  const infer = vi.fn(async (_input: unknown, _options?: unknown) => [labels])
  const dispose = vi.fn(async () => undefined)
  return Object.assign(infer, { dispose })
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => { resolve = res })
  return { promise, resolve }
}

describe('content-safety-guard browser substrate', () => {
  beforeEach(() => {
    vi.mocked(pipeline).mockReset()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('never starts the pipeline until explicitly loaded', () => {
    const runner = new BrowserSubstrateRunner(createModerationSubstrate())
    expect(runner.status).toBe('idle')
    expect(pipeline).not.toHaveBeenCalled()
  })

  it('loads the pinned revision with the caller-chosen device/dtype (default webgpu/q8)', async () => {
    const classifier = makeClassifier(fullScores({ unsafe: 0.8 }))
    vi.mocked(pipeline).mockResolvedValueOnce(classifier as never)
    const runner = new BrowserSubstrateRunner(createModerationSubstrate())
    await runner.load()
    expect(runner.status).toBe('ready')
    expect(pipeline).toHaveBeenCalledWith('text-classification', MODERATION_MODEL_ID, expect.objectContaining({
      device: 'webgpu',
      dtype: 'q8',
      revision: MODERATION_MODEL_REVISION,
    }))
    const result = await runner.run({ text: 'hello' })
    expect(result.kind).toBe('moderation')
    expect(result.unsafe).toBe(0.8)
    expect(Object.keys(result.categories).sort()).toEqual([...MODERATION_CATEGORIES].sort())
    await runner.dispose()
    expect(classifier.dispose).toHaveBeenCalledOnce()
  })

  it('honors an explicit device and dtype override', async () => {
    vi.mocked(pipeline).mockResolvedValueOnce(makeClassifier(fullScores()) as never)
    const runner = new BrowserSubstrateRunner(createModerationSubstrate({ device: 'wasm', dtype: 'fp16' }))
    await runner.load()
    expect(pipeline).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.objectContaining({
      device: 'wasm',
      dtype: 'fp16',
    }))
    await runner.dispose()
  })

  it('judges a prompt/response pair in context by forwarding text_pair', async () => {
    const classifier = makeClassifier(fullScores())
    vi.mocked(pipeline).mockResolvedValueOnce(classifier as never)
    const runner = new BrowserSubstrateRunner(createModerationSubstrate())
    await runner.load()
    await runner.run({ text: 'solo prompt' })
    expect(classifier).toHaveBeenLastCalledWith('solo prompt', { top_k: null })
    await runner.run({ text: 'the prompt', textPair: 'the response' })
    expect(classifier).toHaveBeenLastCalledWith({ text: 'the prompt', text_pair: 'the response' }, { top_k: null })
    await runner.dispose()
  })

  it('rejects empty text before invoking the classifier', async () => {
    const classifier = makeClassifier(fullScores())
    vi.mocked(pipeline).mockResolvedValueOnce(classifier as never)
    const runner = new BrowserSubstrateRunner(createModerationSubstrate())
    await runner.load()
    const before = classifier.mock.calls.length
    await expect(runner.run({ text: '   ' })).rejects.toThrow(/nonempty/)
    expect(classifier).toHaveBeenCalledTimes(before)
    await runner.dispose()
  })

  it('aborts before load and never invokes the pipeline', async () => {
    const controller = new AbortController()
    const runner = new BrowserSubstrateRunner(createModerationSubstrate())
    const loading = runner.load(undefined, controller.signal)
    controller.abort()
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(pipeline).not.toHaveBeenCalled()
    expect(runner.status).toBe('idle')
  })

  it('disposes a classifier that finishes after cancellation rather than publishing it', async () => {
    const pending = deferred<Classifier>()
    vi.mocked(pipeline).mockReturnValueOnce(pending.promise as never)
    const runner = new BrowserSubstrateRunner(createModerationSubstrate())
    const controller = new AbortController()
    const loading = runner.load(undefined, controller.signal)
    await vi.waitFor(() => expect(pipeline).toHaveBeenCalledOnce())
    controller.abort()
    const classifier = makeClassifier(fullScores())
    pending.resolve(classifier)
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' })
    expect(classifier.dispose).toHaveBeenCalledOnce()
    expect(runner.status).toBe('idle')
  })
})
