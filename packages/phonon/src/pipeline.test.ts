import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RuntimeContext } from '@lil-local/inference-core';
import { PhononPipeline } from './pipeline';
import { phononManifest } from './manifest';

const { mockAssetResolve } = vi.hoisted(() => ({ mockAssetResolve: vi.fn() }));

vi.mock('onnxruntime-web/wasm', () => ({
  InferenceSession: {
    create: vi.fn(async () => ({
      run: vi.fn(async () => ({})),
      release: vi.fn(async () => {}),
    })),
  },
}));

function fakeContext(): RuntimeContext {
  return {
    backend: 'wasm',
    assets: {
      resolve: mockAssetResolve,
    } as RuntimeContext['assets'],
    liteRt: {} as RuntimeContext['liteRt'],
  };
}

describe('PhononPipeline', () => {
  const minimalTokenizerJson = JSON.stringify({
    version: '1.0',
    added_tokens: [
      { id: 0, content: '', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
      { id: 1, content: '', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
      { id: 2, content: '', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
      { id: 3, content: '<pad>', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
    ],
    model: {
      type: 'Unigram',
      unk_id: 0,
      vocab: [
        ['', 0.0],
        ['', 0.0],
        ['', 0.0],
        ['<pad>', 0.0],
        ['▁hello', -1.0],
        ['▁world', -1.0],
        ['h', -5.0],
        ['e', -5.0],
        ['l', -5.0],
        ['o', -5.0],
        ['w', -5.0],
        ['r', -5.0],
        ['d', -5.0],
      ],
    },
  });

  beforeEach(() => {
    mockAssetResolve.mockReset();
    mockAssetResolve.mockImplementation((asset) => {
      if (asset.id === 'tokenizer') {
        return Promise.resolve(new TextEncoder().encode(minimalTokenizerJson).buffer);
      }
      return Promise.resolve(new ArrayBuffer(0));
    });
  });

  it('has correct manifest', () => {
    const p = new PhononPipeline();
    expect(p.manifest.modelId).toBe('phonon-31m-en-ONNX');
    expect(p.manifest.capabilities).toContain('text-to-speech');
    expect(p.manifest.backends.wasm).toBe(true);
  });

  it('starts idle', () => {
    const p = new PhononPipeline();
    expect(p.status).toBe('idle');
  });

  it('throws on run before load', async () => {
    const p = new PhononPipeline();
    await expect(p.run({ text: 'hello' })).rejects.toThrow('Pipeline not ready');
  });

  it('loads and becomes ready', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());
    expect(p.status).toBe('ready');
  });

  it('produces a valid AudioInferenceResult after load', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());

    const result = await p.run({ text: 'Hello world' });

    expect(result.kind).toBe('audio');
    expect(result.samples).toBeInstanceOf(Float32Array);
    expect(result.sampleRate).toBe(24000);
    expect(result.channels).toBe(1);
    expect(result.receipt.modelId).toBe('phonon-31m-en-ONNX');
    expect(result.receipt.backend).toBe('wasm');
    expect(result.receipt.inputSummary).toContain('11 characters');
  });

  it('honors an aborted signal on run', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());

    const controller = new AbortController();
    controller.abort();

    await expect(p.run({ text: 'x' }, {}, controller.signal)).rejects.toThrow('CANCELLED');
  });

  it('disposes and moves to disposed status', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());
    await p.dispose();
    expect(p.status).toBe('disposed');
  });

  it('reports load progress', async () => {
    const p = new PhononPipeline();
    const progress: Array<{ phase: string; step: number; total: number }> = [];
    p.onProgress = (prog) => progress.push(prog);

    await p.load(fakeContext());

    expect(progress.length).toBe(5);
    expect(progress[0].phase).toBe('loading');
    expect(progress[4].phase).toBe('loading');
    expect(progress[4].step).toBe(5);
    expect(progress[4].total).toBe(5);
  });
});
