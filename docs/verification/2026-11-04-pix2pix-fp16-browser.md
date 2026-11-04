# Pix2Pix opt-in FP16-packed browser qualification — 2026-11-04

## Scope and integration decision

This is a **real feature in Lil Local**, not a new worktree or standalone
inference runtime: an optional compressed on-disk format for the existing
`@lil-local/pix2pix-webgpu` model. The original FP32 model, manifest, and OPFS
cache identity remain unchanged and are still the default. The new model is
opt-in through **Pix2Pix · Local WebGPU → Model weight format**.

The optimization targets asset download and OPFS storage only. It does **not**
claim GPU-resident FP16 buffers, faster convolutions, higher FPS, or mobile
qualification. WGSL and GPU computation remain FP32.

## Source assets and exact format

- Original folded source: `$HOME/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-v1.bin`
  - Size: **217,677,836** bytes (54,419,459 FP32 values).
  - SHA-256: `eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34`.
- New derived asset: `$HOME/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-fp16-v1.bin`
  - Size: **108,869,644** bytes.
  - SHA-256: `c74e2dbe83c0acb4cb83a8130186485505fb92b3d00b5d1a60463c686b78a978`.
- Conversion: `packages/pix2pix-webgpu/native/pack_fp16.py`.
  Source identity/byte size is validated before export; output cannot
  silently replace an existing file.
- Layout: the same ordered encoder e1-e8 and decoder d1-d8 spans as the
  Rust source. Of 54,419,459 values, **54,404,096** are convolution/deconvolution
  coefficients stored as little-endian FP16. The remaining **15,363**
  biases and folded normalization coefficients remain little-endian FP32.
  No extra format header or padding is stored.
- The `JsUint8WeightReader` validates exact input size and only expands
  convolution spans in bounded <=4 MiB FP32 chunks, uploading to the original
  FP32 GPU buffers. No whole-model FP32 expansion is performed in Wasm.
  The existing FP32 reader/constructor is unchanged in behavior.
- One unique asset ID, version, bytes/hash and independent OPFS key are
  selected by explicit `weightFormat: 'fp16-packed'`; default is `'fp32'`.
  Invalid assets are rejected by existing manifest integrity validation.

## TDD and static verification

- `native/test_pack_fp16.py` was initially RED before the exporter existed,
  then passed 2 tests. Tests cover exact byte accounting and the coefficient-
  versus-affine encoding distinction.
- The pure `native/src/packed.rs` decoder has 3 tests (stored-width
  selection, FP16 scalar expansion/malformed lengths, subnormal/Inf cases)
  passed via native `rustc --test`. Full `cargo test` for the browser crate
  is not available on native Linux because existing
  `wgpu::SurfaceTarget::Canvas` is browser-only.
- Browser-targeted `cargo check --target wasm32-unknown-unknown` passed.
  Release Wasm built with Cargo 1.93.1, then matching
  `wasm-bindgen-cli 0.2.130` regenerated checked-in bindings.
  New artifact SHA-256:
  - `wasm/pix2pix_wgpu_bg.wasm`:
    `c780362d417ffdcc02cbc87a720c02e8f353008f28a39f32254d0adc02ec40a0`.
  - `wasm/pix2pix_wgpu.js`:
    `fb9441c1500ab7b769541368fc7d61146d10e6c0520d8ab1b6ec81228a2ca1fd`.
- The Pix2Pix package's 10 Vitest tests and typecheck passed, including an
  initially RED test verifying packed format selects the distinct constructor.
- Playground storage tests passed, including an initially RED test proving
  FP32 and packed cache identities do not overwrite each other.
- **Full `pnpm verify` passed with exit 0**, including the packed external
  consumer surface, tests, compatibility, conversion asset checks and
  production build. Existing unrelated build chunk/dynamic-import notices
  remained advisory.

## Actual headed browser/UI test

Environment: yuzu-mint, Chrome Beta 158, Linux Mint X11, explicit WebGPU
developer/Vulkan flags, `adapter.info.vendor = 'nvidia'`,
`adapter.info.architecture = 'turing'`, RTX 2070 SUPER.

Test ran the **real Lil Local Vite application UI**, not a test-only model
implementation. In a fresh Playwright browser context:

1. Opened the existing Pix2Pix workspace from the model catalog.
2. Chose **Packed FP16 convolutions (103.8 MiB)**.
3. Selected the 108,869,644-byte derived `.bin`, imported it through the
   production integrity-verified resolver into the distinct OPFS key, and
   observed **Installed in browser storage**.
4. Clicked **Load installed model**: existing Rust/Wasm runtime initialized
   through the new packed reader, with the pinned BatchNorm-eval/skip settings.
5. Loaded the 256×256 left edge half of a publicly available paired shoe
   image and clicked **Run Pix2Pix**.
6. Observed a recognizable generated shoe, a normal session receipt and
   **zero page/console errors**. The screenshot was left locally at
   `/tmp/lil-pix2pix-weight-compare-20261104/app-packed-fp16.png`.
7. Example session receipt: 248.8 ms load, 377.8 ms compile, 49.1 ms
   inference+readback. These are **single observations, not benchmarks**.

## Existing FP32 UI regression smoke

A second fresh headed Chrome Beta browser context exercised the **default
FP32 format** using the same production model-list → import → OPFS → load →
run controls and the same shoe-edge image. It also completed end-to-end with a
recognizable generated shoe, normal session receipt and **zero page/console
errors**. This checks that existing users' path was preserved, not just that
the new packed constructor works. Screenshot:
`/tmp/lil-pix2pix-weight-compare-20261104/app-original-fp32.png`.

## Actual FP32-vs-FP16 image parity

A separate headed Playwright/Chrome Beta run used the **same compiled checked-in
Wasm module, real GPU adapter, same normalized 256×256 shoe-edge input**,
original FP32 and packed-FP16 model artifacts. It invoked the two separate
Wasm constructors and actual `run_frame()/read_output_nchw()` calls.

- Original bytes fetched: **217,677,836**.
- Packed bytes fetched: **108,869,644**.
- Mean absolute FP32 output delta: **0.00051356725**.
- Maximum absolute FP32 output delta: **0.00811781**.
- Mean absolute RGB8 channel difference: **0.06548055 / 255**.
- Maximum RGB8 channel difference: **1 / 255**.
- RGB channels within 1 intensity level: **100%**.
- Browser page/console errors: **zero**. Process exited 0 after numerical
  tolerance assertion. Temporary harness:
  `/tmp/lil-pix2pix-weight-compare-20261104/compare-packed.html`
  with `check-packed.cjs` (not a production dependency).

A separate November 4 scouting pass had already assessed the effect of
reducing precision on the same real shoe model by re-expanding variant files
to FP32. The new packed-reader browser result **matches** the independently
observed all-convolution FP16 output error (0.06548055 RGB8), supporting
correct span order and decoding.

## Boundaries and user-facing result

- Real compressed file and browser OPFS storage reduction are **verified**;
  GPU memory/throughput improvements are **not**.
- This is Pix2Pix edge-to-shoe translation, **not diffusion**, neural
  symbolic replacement, generalized model compression, or evidence of
  real-time full-resolution diffusion on a phone.
- Quality checked on one paired shoe input, with earlier synthetic examples
  as supplemental numerical checks. Broad dataset/mobile/GPU qualification
  is still open.
- Source model weights are not committed or bundled; the original verified
  FP32 model remains unmodified and is the default option.
- This is worth integrating into Lil Local because it cuts a real existing
  model's installation size in half with measured output parity. It does
  **not** justify integration of the old tiny `wgpu-vae-lab` into Lil Local.
