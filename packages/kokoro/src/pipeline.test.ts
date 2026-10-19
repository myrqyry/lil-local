import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeContext } from '@lil-local/inference-core';
import { KokoroPipeline } from "./pipeline";

const { mockGenerate } = vi.hoisted(() => ({ mockGenerate: vi.fn() }));

vi.mock("kokoro-js", () => ({
  KokoroTTS: {
    from_pretrained: vi.fn(async () => ({
      generate: mockGenerate,
      list_voices: () => {},
    })),
  },
}));

function fakeContext(): RuntimeContext {
  return {
    backend: "wasm",
    assets: {} as RuntimeContext["assets"],
    liteRt: {} as RuntimeContext["liteRt"],
  };
}

describe("KokoroPipeline", () => {
  beforeEach(() => {
    mockGenerate.mockReset();
    mockGenerate.mockResolvedValue({
      audio: new Float32Array(24000).fill(0.5),
      sampling_rate: 24000,
    });
  });

  it("throws on run before load", async () => {
    const p = new KokoroPipeline();
    await expect(p.run({ text: "hello" })).rejects.toThrow("Pipeline not ready");
  });

  it("has correct manifest", () => {
    const p = new KokoroPipeline();
    expect(p.manifest.modelId).toBe("kokoro-82m-v1.0");
    expect(p.manifest.capabilities).toContain("text-to-speech");
  });

  it("starts idle", () => {
    const p = new KokoroPipeline();
    expect(p.status).toBe("idle");
  });

  it("produces a sane AudioInferenceResult via checkAudioValid", async () => {
    const p = new KokoroPipeline();
    await p.load(fakeContext());

    const result = await p.run(
      { text: "Hello world" },
      { voice: "af_heart", speed: 1 }
    );

    expect(result.kind).toBe("audio");
    expect(result.samples).toBeInstanceOf(Float32Array);
    expect(result.samples.length).toBe(24000);
    expect(result.sampleRate).toBe(24000);
    expect(result.channels).toBe(1);
    expect(result.durationSeconds).toBeCloseTo(1, 3);
    expect(result.receipt.modelId).toBe("kokoro-82m-v1.0");
    expect(result.receipt.backend).toBe("wasm");
    expect(result.receipt.inputSummary).toBe("11 characters");
    expect(result.receipt.outputSummary).toContain(
      "24000 samples at 24000Hz"
    );
    expect(result.receipt.warnings).toEqual([]);
    expect(mockGenerate).toHaveBeenCalledWith("Hello world", {
      voice: "af_heart",
      speed: 1,
    });
  });

  it("honors an aborted signal", async () => {
    const p = new KokoroPipeline();
    await p.load(fakeContext());

    const controller = new AbortController();
    controller.abort();

    await expect(
      p.run({ text: "x" }, {}, controller.signal)
    ).rejects.toThrow("CANCELLED");
  });
});
