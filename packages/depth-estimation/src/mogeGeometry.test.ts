import {describe, expect, it} from 'vitest';

import {decodeMoGeOutputs, MOGE_PIXELS, MOGE_SIZE, prepareMoGeImage, moGeGeometryToDepthFrame} from './mogeGeometry';

const fixture = () => {
  const normals = new Float32Array(MOGE_PIXELS * 3);
  const points = new Float32Array(MOGE_PIXELS * 3);
  const mask = new Float32Array(MOGE_PIXELS);
  for (let i = 0; i < MOGE_PIXELS; i += 1) {
    normals[i * 3 + 2] = 1;
    points[i * 3 + 2] = 2 + i / MOGE_PIXELS;
    mask[i] = i < MOGE_PIXELS / 2 ? 0.99 : 0.2;
  }
  return [
    {shape: [1, MOGE_SIZE, MOGE_SIZE, 3], values: normals},
    {shape: [1, 1, 1, 1], values: new Float32Array([0.75])},
    {shape: [1, MOGE_SIZE, MOGE_SIZE, 3], values: points},
    {shape: [1, MOGE_SIZE, MOGE_SIZE, 1], values: mask},
  ];
};

const meta = {
  modelId: 'litert-community/MoGe-2-LiteRT',
  modelRevision: 'immutable-revision',
  backend: 'webgpu' as const,
  sourceWidth: 640,
  sourceHeight: 360,
  inferenceMs: 122,
  frameId: 'camera-1',
};

describe('MoGe-2 geometry contract', () => {
  it('preprocesses NCHW float32 RGB in [0,1] without channel mixing', () => {
    const image = {
      width: 1,
      height: 1,
      data: new Uint8ClampedArray([255, 128, 0, 255]),
    } as ImageData;
    const input = prepareMoGeImage(image);
    expect(input.length).toBe(MOGE_PIXELS * 3);
    expect(input[0]).toBe(1);
    expect(input[MOGE_PIXELS]).toBeCloseTo(128 / 255, 6);
    expect(input[2 * MOGE_PIXELS]).toBe(0);
    expect(input[MOGE_PIXELS - 1]).toBe(1);
  });

  it('bilinearly resamples image content while retaining RGB plane order', () => {
    const image = {
      width: 2, height: 1,
      data: new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 255]),
    } as ImageData;
    const input = prepareMoGeImage(image);
    const mid = MOGE_SIZE / 2;
    expect(input[0]).toBe(1);
    expect(input[MOGE_PIXELS - 1]).toBe(0);
    expect(input[mid]).toBeLessThan(0.6);
    expect(input[mid]).toBeGreaterThan(0.4);
    expect(input[2 * MOGE_PIXELS + mid]).toBeGreaterThan(0.4);
  });

  it('identifies both three-channel heads by normal magnitude, not tensor order', () => {
    const values = fixture();
    const decoded = decodeMoGeOutputs([values[2], values[3], values[0], values[1]], meta);
    expect(decoded.normals).toBe(values[0].values);
    expect(decoded.affinePoints).toBe(values[2].values);
    expect(decoded.validFraction).toBe(0.5);
    expect(decoded.relativeDepth[0]).toBe(0);
    expect(decoded.relativeDepth[Math.floor(MOGE_PIXELS / 2) - 1]).toBeCloseTo(1, 5);
    expect(decoded.relativeDepth[MOGE_PIXELS - 1]).toBe(0);
    expect(decoded.scaleFactor).toBe(0.75);
    expect(decoded.frameId).toBe('camera-1');
    const canonical = moGeGeometryToDepthFrame(decoded, 12345);
    expect(canonical).toMatchObject({kind: 'depth', frameId: 'camera-1',
      width: 448, height: 448, sourceWidth: 640, sourceHeight: 360,
      timestampMs: 12345, normalization: 'frame-minmax', nearFarConvention: 'higher-far'});
    expect(canonical.data).not.toBe(decoded.relativeDepth);
    expect(canonical.data[1000]).toBeCloseTo(decoded.relativeDepth[1000]);
    expect(decoded.sourceToModelUV.scaleX).toBe(1);
  });

  it('rejects mismatched outputs and ambiguous geometry instead of fabricating depth', () => {
    const outputs = fixture();
    expect(() => decodeMoGeOutputs(outputs.slice(1), meta)).toThrow('four outputs');
    outputs[0].values.fill(0);
    expect(() => decodeMoGeOutputs(outputs, meta)).toThrow('Cannot reliably distinguish');
    outputs[0].values.fill(1);
    outputs[1].values[0] = Number.NaN;
    expect(() => decodeMoGeOutputs(outputs, meta)).toThrow();
  });

  it('rejects invalid input image dimensions', () => {
    expect(() => prepareMoGeImage({
      width: 2, height: 2, data: new Uint8ClampedArray(4),
    } as ImageData)).toThrow('nonempty RGBA');
  });
});
