/* tslint:disable */
/* eslint-disable */

export enum DropoutConvention {
    /**
     * Original Torch7 (`phillipi/pix2pix/models.lua`) convention: 50% dropout on d1, d2, d3.
     */
    Torch7D1ToD3 = 0,
    /**
     * PyTorch (`junyanz/pytorch-CycleGAN-and-pix2pix`) convention: 50% dropout on d2, d3, d4.
     */
    PyTorchD2ToD4 = 1,
    /**
     * Deterministic inference with test-time dropout disabled on all layers.
     */
    Disabled = 2,
}

export enum NormalizationConvention {
    /**
     * Compute spatial InstanceNorm statistics from each current activation.
     */
    InstanceNorm = 0,
    /**
     * Apply export-folded BatchNorm eval as per-channel `x * scale + shift`.
     */
    BatchNormEvalFolded = 1,
}

export class Pix2PixEngine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Executes the on-device WGSL `compute_mse` verification kernel comparing the GPU's
     * most recent `d8` output tensor (`[3, 256, 256]`) against a CPU reference tensor,
     * returning `[mse, rmse, max_abs_err, psnr_db]`.
     */
    compute_on_device_mse(cpu_ref_nchw: Float32Array | number[]): Promise<Float32Array>;
    /**
     * Creates the WebGPU Pix2Pix U-Net 256 engine from a borrowed byte slice.
     */
    static create(canvas: HTMLCanvasElement, weights_bin: Uint8Array | number[]): Promise<Pix2PixEngine>;
    /**
     * Reads packed FP16 convolution coefficients and FP32 bias/normalization
     * spans from a verified asset, expanding only a bounded chunk at a time.
     * The original FP32 WebGPU compute kernels and buffer layouts are unchanged.
     */
    static create_from_packed_fp16_uint8_array(canvas: HTMLCanvasElement, weights_array: Uint8Array, dropout_convention: DropoutConvention): Promise<Pix2PixEngine>;
    /**
     * Creates the WebGPU Pix2Pix U-Net 256 engine by streaming 4 MiB chunks directly from
     * a `js_sys::Uint8Array`, eliminating the 217.68 MB Wasm linear memory allocation spike.
     */
    static create_from_uint8_array(canvas: HTMLCanvasElement, weights_array: Uint8Array, dropout_convention: DropoutConvention): Promise<Pix2PixEngine>;
    /**
     * Read back the most recent d8 output tensor in NCHW order ([3, 256, 256]).
     *
     * This is intentionally separate from run_frame(): callers that only render to
     * the canvas avoid a full 768 KiB GPU->CPU transfer.
     */
    read_output_nchw(): Promise<Float32Array>;
    run_frame(rgba_bytes: Uint8Array | number[]): void;
    /**
     * Dynamically switches the active dropout layer placement convention (`d1..d3`, `d2..d4`, or `Disabled`).
     */
    set_dropout_convention(convention: DropoutConvention): void;
    /**
     * Switches normalization semantics across all normalized encoder/decoder stages.
     *
     * `BatchNormEvalFolded` requires weights exported with running-statistics
     * folded into the stored per-channel gamma/beta slots.
     */
    set_normalization_convention(convention: NormalizationConvention): void;
    /**
     * Toggles hardware-level `[skip, decoder]` vs `[decoder, skip]` channel order bit (bit 2 = 4u)
     * across all skip-connected decoder layers (`d2..d8`).
     */
    set_raw_pytorch_skip_order(raw_skip_first: boolean): void;
    /**
     * Toggles between dual-storage-buffer skip concatenation (`false`) and contiguous
     * `Texture2DArray` (`rgba32float`) skip-feature stacking (`true`) on `d2..d8`.
     */
    set_use_texture2d_array_skip(enabled: boolean): void;
    /**
     * Toggles between 2-Pass Centered FP32 (`false`) and 1-Pass Welford + Chan Tree Reduction (`true`)
     * across all encoder and decoder InstanceNorm layers.
     */
    set_use_welford(use_welford: boolean): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_pix2pixengine_free: (a: number, b: number) => void;
    readonly pix2pixengine_compute_on_device_mse: (a: number, b: number, c: number) => any;
    readonly pix2pixengine_create: (a: any, b: number, c: number) => any;
    readonly pix2pixengine_create_from_packed_fp16_uint8_array: (a: any, b: any, c: number) => any;
    readonly pix2pixengine_create_from_uint8_array: (a: any, b: any, c: number) => any;
    readonly pix2pixengine_read_output_nchw: (a: number) => any;
    readonly pix2pixengine_run_frame: (a: number, b: number, c: number) => [number, number];
    readonly pix2pixengine_set_dropout_convention: (a: number, b: number) => void;
    readonly pix2pixengine_set_normalization_convention: (a: number, b: number) => void;
    readonly pix2pixengine_set_raw_pytorch_skip_order: (a: number, b: number) => void;
    readonly pix2pixengine_set_use_texture2d_array_skip: (a: number, b: number) => void;
    readonly pix2pixengine_set_use_welford: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1ef2b1b30d297ab2: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h5cb34573169467c3: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h06f2931b8d7cfd15: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h06f2931b8d7cfd15_14: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
