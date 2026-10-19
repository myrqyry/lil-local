import { describe, expect, it, beforeEach, vi } from 'vitest';
import type { RuntimeContext, LiteRtRuntime, AssetResolver } from '@lil-local/inference-core';

vi.mock('../codec', () => ({
  CodecDecoder: class {
    decode = vi.fn().mockResolvedValue(new Float32Array([0, 0.1, 0.2]));
  },
}));

import { DecoderPhase } from './decoder';
import { qwen3TtsVariants } from '../manifest';
import { packCodecFrames } from '../codec-frames';

function fakeLiteRt() {
  const inputDetails = [{ name: 'args_0', shape: [1, 16, 64] }];
  const liteRt: LiteRtRuntime = {
    loadModel: vi.fn().mockResolvedValue({
      signatures: { decode: { getInputDetails: () => inputDetails } },
      getInputDetails: () => inputDetails,
    }),
    loadNpy: vi.fn(),
    fetchBuffer: vi.fn(),
  };
  return liteRt;
}

function fakeContext(liteRt: LiteRtRuntime): RuntimeContext {
  const assets: AssetResolver = { resolve: vi.fn() };
  return { backend: 'wasm' as const, assets, liteRt };
}

describe('DecoderPhase', () => {
  let phase: DecoderPhase;
  beforeEach(() => {
    phase = new DecoderPhase(qwen3TtsVariants.int4);
  });

  it('compiles only the codec graph', async () => {
    const liteRt = fakeLiteRt();
    await phase.load(fakeContext(liteRt));
    expect(liteRt.loadModel).toHaveBeenCalledWith('codec_decoder_fp32.tflite');
    expect(phase.name).toBe('decoder');
  });

  it('decodes CodecFrames into a Float32Array', async () => {
    const liteRt = fakeLiteRt();
    await phase.load(fakeContext(liteRt));
    const audio = await phase.decode(packCodecFrames([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]));
    expect(audio).toBeInstanceOf(Float32Array);
    expect(audio.length).toBeGreaterThan(0);
  });
});
