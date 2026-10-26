import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RuntimeContext } from '@lil-local/inference-core';
import { PhononPipeline } from './pipeline';
import { phononManifest } from './manifest';

const mocks = vi.hoisted(() => ({
  resolve: vi.fn(), create: vi.fn(), flow: vi.fn(), decoder: vi.fn(),
  invalidOutputs: false,
}));
vi.mock('@huggingface/transformers', () => ({
  AutoTokenizer: { from_pretrained: vi.fn(async () => ({
    encode: (_text: string, opts: unknown) => {
      if ((opts as { add_special_tokens?: boolean })?.add_special_tokens !== false)
        throw new Error('Phonon tokenizer must not insert special tokens');
      return [11, 12, 13];
    },
  })) },
}));
vi.mock('onnxruntime-web/wasm', () => ({
  Tensor: class MockTensor {
    readonly location = 'cpu';
    constructor(readonly type: string, readonly data: ArrayLike<number | bigint>, readonly dims: number[]) {}
  },
  InferenceSession: { create: mocks.create },
}));
function tensor(data: Float32Array | BigInt64Array, dims: number[]) {
  return { data, dims, location: 'cpu', dispose: vi.fn() };
}
const flowResult = () => ({
  latent_out: tensor(new Float32Array(32).fill(.15), [1, 1, 32]),
  eos_logit: tensor(new Float32Array([10]), [1, 1]),
  ...Object.fromEntries(Array.from({ length: 6 }, (_, i) => [
    ['present.' + i + '.key', tensor(new Float32Array(512), [1, 8, 1, 64])],
    ['present.' + i + '.value', tensor(new Float32Array(512), [1, 8, 1, 64])],
  ]).flat()),
});
const decoderResult = () => ({
  audio: tensor(Float32Array.from({ length: 1920 }, (_, i) => Math.sin(i / 20) * .08), [1, 1, 1920]),
  present_offset: tensor(BigInt64Array.of(1n), [1]),
  ...Object.fromEntries(Array.from({ length: 2 }, (_, i) => [
    ['present.' + i + '.key', tensor(new Float32Array(512), [1, 8, 1, 64])],
    ['present.' + i + '.value', tensor(new Float32Array(512), [1, 8, 1, 64])],
  ]).flat()),
});

function fakeContext(): RuntimeContext {
  return {
    backend: 'wasm',
    assets: { resolve: mocks.resolve } as RuntimeContext['assets'],
    liteRt: {} as RuntimeContext['liteRt'],
  };
}

describe('Phonon actual ONNX graph boundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.invalidOutputs = false;
    mocks.resolve.mockImplementation(async () => new Float32Array(512).buffer);
    mocks.flow.mockImplementation(async () => mocks.invalidOutputs ? {} : flowResult());
    mocks.decoder.mockImplementation(async () => mocks.invalidOutputs ? {} : decoderResult());
    mocks.create.mockImplementation(async (path: string) => {
      const flow = path.includes('flow_lm');
      return {
        inputNames: flow ? ['text_ids'] : ['latent'],
        outputNames: flow ? ['latent_out'] : ['audio'],
        run: flow ? mocks.flow : mocks.decoder,
        release: vi.fn(async () => {}),
      };
    });
  });

  it('declares the real text-to-speech capability', () => {
    const p = new PhononPipeline();
    expect(p.manifest.modelId).toBe('phonon-31m-en-ONNX');
    expect(phononManifest.capabilities).toContain('text-to-speech');
  });

  it('refuses run before an explicit model load', async () => {
    await expect(new PhononPipeline().run({ text: 'hello' })).rejects.toThrow(/not loaded/);
  });

  it('loads the matched real ONNX graph set, with separate sidecar metadata', async () => {
    const p = new PhononPipeline({ variant: 'q4' });
    await p.load(fakeContext());
    expect(p.status).toBe('ready');
    expect(p.manifest.assets.find(asset => asset.id === 'flow-lm')?.path).toBe('onnx/flow_lm_q4.onnx');
    expect(mocks.create).toHaveBeenCalledTimes(2);
    expect(mocks.create.mock.calls[0][0]).toContain('flow_lm_q4.onnx');
    expect(mocks.create.mock.calls[0][1].externalData[0].path).toBe('flow_lm_q4.onnx_data');
    expect(mocks.create.mock.calls[1][0]).toContain('mimi_decoder_q4.onnx');
    expect(mocks.create.mock.calls[1][1].externalData).toBeUndefined();
    await p.dispose();
  });

  it('rejects absent graph outputs instead of returning a fabricated 80ms silent frame', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());
    mocks.invalidOutputs = true;
    await expect(p.run({ text: 'Hello world' })).rejects.toThrow(/latent_out/);
    expect(mocks.flow).toHaveBeenCalled();
    expect(mocks.decoder).not.toHaveBeenCalled();
    await p.dispose();
  });

  it('executes both graphs and returns nonzero PCM from the decoder, including streaming callbacks', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());
    const onChunk = vi.fn();
    const result = await p.run({ text: 'Hello world' }, { temperature: 0, onChunk });
    expect(mocks.flow.mock.calls.length).toBeGreaterThan(2);
    expect(mocks.decoder.mock.calls.length).toBeGreaterThan(0);
    const feeds = mocks.flow.mock.calls[2][0];
    expect(feeds).toHaveProperty('text_ids');
    expect(feeds).toHaveProperty('past_key_values.0.key');
    expect(feeds).toHaveProperty('latent');
    expect(result.kind).toBe('audio');
    expect(result.sampleRate).toBe(24000);
    expect(result.samples.length).toBe(1920 * mocks.decoder.mock.calls.length);
    expect(result.samples.some(x => Math.abs(x) > .001)).toBe(true);
    expect(onChunk).toHaveBeenCalledTimes(mocks.decoder.mock.calls.length);
    expect(result.receipt.modelId).toBe('phonon-31m-en-ONNX');
    await p.dispose();
  });

  it('rejects empty text, invalid temperature, and aborted generation', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());
    await expect(p.run({ text: '' })).rejects.toThrow(/empty/);
    await expect(p.run({ text: 'hi' }, { temperature: -1 })).rejects.toThrow(/temperature/);
    const abort = new AbortController();
    abort.abort();
    await expect(p.run({ text: 'hello' }, undefined, abort.signal)).rejects.toThrow('CANCELLED');
    await p.dispose();
  });

  it('refuses to run q4f16 on WASM rather than silently choosing an incompatible model', async () => {
    await expect(new PhononPipeline({ variant: 'q4f16' }).load(fakeContext()))
      .rejects.toThrow(/requires WebGPU/);
  });

  it('disposes both ONNX sessions', async () => {
    const p = new PhononPipeline();
    await p.load(fakeContext());
    await p.dispose();
    expect(p.status).toBe('disposed');
    await expect(p.run({ text: 'hello' })).rejects.toThrow(/not loaded/);
  });
});

import { choosePhononVariant } from './variant';

describe('Phonon backend-to-graph selection', () => {
  it('uses fp32 for WASM, q4 for WebGPU lacking shader-f16, and q4f16 when supported', () => {
    expect(choosePhononVariant('wasm', false)).toBe('fp32');
    expect(choosePhononVariant('webgpu', false)).toBe('q4');
    expect(choosePhononVariant('webgpu', true)).toBe('q4f16');
  });
  it('never silently selects explicitly requested incompatible fp16 graphs', () => {
    expect(() => choosePhononVariant('webgpu', false, 'q4f16')).toThrow(/shader-f16/);
    expect(() => choosePhononVariant('wasm', false, 'q4f16')).toThrow(/WebGPU/);
  });
});

import { createPhononSubstrate } from './index';
import { BrowserSubstrateRunner } from '@lil-local/inference-core';

describe('Phonon engine-neutral consumer substrate', () => {
  it('constructs inertly without requiring an unrelated LiteRT instance', async () => {
    mocks.create.mockClear();
    const substrate = createPhononSubstrate({
      backend: 'wasm',
      assets: { resolve: mocks.resolve },
      variant: 'q4',
    });
    expect(substrate.format).toBe('onnx');
    expect(substrate.capabilities).toEqual(['text-to-speech']);
    expect(substrate.requestedBackends).toEqual(['wasm']);
    const runner = new BrowserSubstrateRunner(substrate);
    expect(runner.status).toBe('idle');
    expect(mocks.create).not.toHaveBeenCalled();
    await runner.dispose();
  });
});
