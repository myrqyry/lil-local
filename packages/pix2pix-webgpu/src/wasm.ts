export interface Pix2PixWasmEngine {
  set_normalization_convention(convention: number): void
  set_raw_pytorch_skip_order(rawSkipFirst: boolean): void
  set_use_texture2d_array_skip(enabled: boolean): void
  run_frame(rgbaBytes: Uint8Array): void
  read_output_nchw(): Promise<Float32Array>
  free(): void
}

export interface Pix2PixWasmRuntimeModule {
  Pix2PixEngine: {
    create_from_uint8_array(
      canvas: HTMLCanvasElement,
      weights: Uint8Array,
      dropoutConvention: number,
    ): Promise<Pix2PixWasmEngine>
  }
  DropoutConvention: { Disabled: number }
  NormalizationConvention: { BatchNormEvalFolded: number }
}

export async function loadBundledPix2PixWasm(): Promise<Pix2PixWasmRuntimeModule> {
  const module = await import('../wasm/pix2pix_wgpu.js')
  await module.default()
  return module as unknown as Pix2PixWasmRuntimeModule
}
