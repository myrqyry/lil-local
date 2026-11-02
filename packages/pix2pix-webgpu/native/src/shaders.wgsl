// ============================================================================
// Pix2Pix U-Net 256 — Production WebGPU (WGSL) Compute & Verification Shaders
// ============================================================================
// Core Architecture & Performance Guarantees:
// 1. 16x16 Block Tiling (`@workgroup_size(16, 16, 1)`):
//    All spatial and reduction compute kernels execute in 16x16 (256-thread)
//    workgroups to maximize L1/L2 texture and buffer cache locality on modern GPUs.
// 2. Explicit 16-Byte Memory Alignment (`@align(16)`):
//    All Uniform Buffer Object (UBO) and storage structures are padded to 16-byte
//    boundaries and decorated with `@align(16)` to prevent out-of-bounds reads
//    and satisfy strict WebGPU uniform layout constraints.
// 3. Contiguous `Texture2DArray` Skip-Connection Stacking (`stack_skip_features`):
//    Stacks encoder skip activations and decoder activations into a contiguous
//    `texture_storage_2d_array<rgba32float, write>` / `texture_2d_array<f32>`
//    (packing 4 channels per RGBA32Float layer so up to 1024 channels fit within
//    WebGPU's 256-layer `maxTextureArrayLayers` limit) and indexes directly into
//    concatenation depth in `conv_transpose2d_up_texarray` and
//    `conv_transpose2d_up_texarray_storage`.
// 4. Hardware-Accelerated `fast_tanh()` and `clamp()` Built-ins:
//    Replaces scalar exponential approximations with hardware-accelerated
//    `tanh(clamp(x, -10.0, 10.0))` and `clamp()` built-ins.
// 5. On-Device MSE Verification Kernel (`compute_mse`):
//    Computes Mean Squared Error (MSE), RMSE, Max Absolute Error, and PSNR (dB)
//    between the WebGPU final layer output (`d8`) and the CPU/TF.js reference
//    tensor directly on the GPU in a dedicated compute pass.
// ============================================================================

struct LayerParams {
    @align(16) h_in: u32,
    w_in: u32,
    c_in: u32,
    h_out: u32,
    @align(16) w_out: u32,
    c_out: u32,
    // Bit 0 (1u): pre-LeakyReLU(0.2)
    // Bit 1 (2u): pre-ReLU(0.0) via clamp()
    // Bit 2 (4u): raw PyTorch [skip, decoder] weight channel order (when not pre-swapped)
    // Bit 3 (8u): post-fast_tanh()
    flags: u32,
    has_skip: u32,
};

struct NormParams {
    @align(16) h: u32,
    w: u32,
    c: u32,
    apply_dropout: u32,
    @align(16) seed: u32,
    eps: f32,
    use_welford: u32,
    // 0u: spatial InstanceNorm; 1u: folded BatchNorm eval affine (x * gamma + beta)
    mode: u32,
};

struct SkipStackParams {
    @align(16) h: u32,
    w: u32,
    c_in: u32,
    skip_first: u32,
    @align(16) total_c: u32,
    rgba_layers: u32,
    flags: u32,
    _pad0: u32,
};

struct MseParams {
    @align(16) total_elements: u32,
    width: u32,
    height: u32,
    channels: u32,
};

struct MseStatsOut {
    @align(16) mse: f32,
    rmse: f32,
    max_abs_err: f32,
    psnr_db: f32,
};

// UBO weight containers (fits within WebGPU's 64 KiB `maxUniformBufferBindingSize`)
struct ConvUboWeightsE1 {
    @align(16) weights: array<vec4<f32>, 96>, // 8 * 3 * 16 = 384 f32s (1,536 B)
    @align(16) bias: array<vec4<f32>, 2>,     // 8 f32s (32 B)
};

struct DeconvUboWeightsD8 {
    @align(16) weights: array<vec4<f32>, 192>, // 3 * 16 * 16 = 768 f32s (3,072 B)
    @align(16) bias: vec4<f32>,                // 3 f32s padded to 4-float boundary (16 B)
};

// ----------------------------------------------------------------------------
// Hardware-Accelerated Normalization & Activation Built-ins (`fast_tanh`, `clamp`)
// ----------------------------------------------------------------------------
fn fast_tanh(x: f32) -> f32 {
    return tanh(clamp(x, -10.0, 10.0));
}

fn apply_pre_act(val: f32, flags: u32) -> f32 {
    if ((flags & 1u) != 0u) {
        let pos = clamp(val, 0.0, 65504.0);
        let neg = clamp(val, -65504.0, 0.0) * 0.2;
        return pos + neg;
    }
    if ((flags & 2u) != 0u) {
        return clamp(val, 0.0, 65504.0);
    }
    return val;
}

// ----------------------------------------------------------------------------
// Pass 1: Image Preprocessing (RGBA8Unorm Texture -> NCHW [-1.0, 1.0] Buffer)
// ----------------------------------------------------------------------------
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pre_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 256u || gid.y >= 256u) {
        return;
    }
    let px = textureLoad(input_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let idx = gid.y * 256u + gid.x;
    let plane = 65536u; // 256 * 256
    pre_out[0u * plane + idx] = clamp(px.r * 2.0 - 1.0, -1.0, 1.0);
    pre_out[1u * plane + idx] = clamp(px.g * 2.0 - 1.0, -1.0, 1.0);
    pre_out[2u * plane + idx] = clamp(px.b * 2.0 - 1.0, -1.0, 1.0);
}

// ----------------------------------------------------------------------------
// Pass 2A: Encoder Downsampling Conv2d (16x16 Block Tiling, Storage Weights)
// Supports both 3D spatial-tile dispatches `(ceil(w/16), ceil(h/16), c_out)`
// and 1D coalesced dispatches `(ceil(total_out/256), 1, 1)` for small maps.
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> conv_params: LayerParams;
@group(0) @binding(1) var<storage, read> conv_in: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weights: array<f32>;
@group(0) @binding(3) var<storage, read> conv_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> conv_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn conv2d_down(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let p = conv_params;
    var ox: u32 = 0u;
    var oy: u32 = 0u;
    var oc: u32 = 0u;
    var idx: u32 = 0u;

    if (nwg.z == p.c_out && (nwg.y > 1u || p.h_out <= 16u)) {
        ox = gid.x;
        oy = gid.y;
        oc = gid.z;
        if (ox >= p.w_out || oy >= p.h_out || oc >= p.c_out) {
            return;
        }
        idx = (oc * p.h_out + oy) * p.w_out + ox;
    } else {
        idx = wgid.x * 256u + lid.y * 16u + lid.x;
        let total_out = p.c_out * p.h_out * p.w_out;
        if (idx >= total_out) {
            return;
        }
        ox = idx % p.w_out;
        let tmp = idx / p.w_out;
        oy = tmp % p.h_out;
        oc = tmp / p.h_out;
    }

    var sum: f32 = conv_bias[oc];
    let in_plane = p.h_in * p.w_in;
    let w_oc_base = oc * (p.c_in * 16u);
    let base_iy = i32(oy * 2u) - 1;
    let base_ix = i32(ox * 2u) - 1;

    // Active-window loop bounds: prunes out-of-bounds zero-padding taps upfront
    // (cuts bottleneck e8 2x2->1x1 inner loop from 16 taps to the 4 valid center taps)
    let ky_start = u32(max(0, -base_iy));
    let ky_end = u32(clamp(i32(p.h_in) - base_iy, 0, 4));
    let kx_start = u32(max(0, -base_ix));
    let kx_end = u32(clamp(i32(p.w_in) - base_ix, 0, 4));

    for (var ic: u32 = 0u; ic < p.c_in; ic = ic + 1u) {
        let in_c_base = ic * in_plane;
        let w_ic_base = w_oc_base + ic * 16u;

        for (var ky: u32 = ky_start; ky < ky_end; ky = ky + 1u) {
            let iy = u32(base_iy + i32(ky));
            let in_row_base = in_c_base + iy * p.w_in;
            let w_row_base = w_ic_base + ky * 4u;

            for (var kx: u32 = kx_start; kx < kx_end; kx = kx + 1u) {
                let ix = u32(base_ix + i32(kx));
                let raw = conv_in[in_row_base + ix];
                let a = apply_pre_act(raw, p.flags);
                let w = conv_weights[w_row_base + kx];
                sum = sum + a * w;
            }
        }
    }

    conv_out[idx] = sum;
}

// ----------------------------------------------------------------------------
// Pass 2B: UBO-Accelerated Encoder Conv2d (`conv2d_down_ubo`, 16x16 Block Tiling)
// Keeps both `LayerParams` and kernel weights inside Uniform Buffer Objects (UBOs).
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> conv_ubo_params: LayerParams;
@group(0) @binding(1) var<storage, read> conv_ubo_in: array<f32>;
@group(0) @binding(2) var<uniform> conv_ubo_weights: ConvUboWeightsE1;
@group(0) @binding(3) var<storage, read_write> conv_ubo_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn conv2d_down_ubo(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = conv_ubo_params;
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;
    if (ox >= p.w_out || oy >= p.h_out || oc >= p.c_out) {
        return;
    }

    let bias_vec = conv_ubo_weights.bias[oc >> 2u];
    var sum: f32 = bias_vec[oc & 3u];
    let in_plane = p.h_in * p.w_in;
    let base_iy = i32(oy * 2u) - 1;
    let base_ix = i32(ox * 2u) - 1;

    let ky_start = u32(max(0, -base_iy));
    let ky_end = u32(clamp(i32(p.h_in) - base_iy, 0, 4));
    let kx_start = u32(max(0, -base_ix));
    let kx_end = u32(clamp(i32(p.w_in) - base_ix, 0, 4));

    for (var ic: u32 = 0u; ic < p.c_in; ic = ic + 1u) {
        let in_c_base = ic * in_plane;
        let w_vec_base = (oc * p.c_in + ic) * 4u;

        for (var ky: u32 = ky_start; ky < ky_end; ky = ky + 1u) {
            let iy = u32(base_iy + i32(ky));
            let in_row_base = in_c_base + iy * p.w_in;
            let w_row_vec = conv_ubo_weights.weights[w_vec_base + ky];

            for (var kx: u32 = kx_start; kx < kx_end; kx = kx + 1u) {
                let ix = u32(base_ix + i32(kx));
                let raw = conv_ubo_in[in_row_base + ix];
                let a = apply_pre_act(raw, p.flags);
                sum = sum + a * w_row_vec[kx];
            }
        }
    }

    let out_idx = (oc * p.h_out + oy) * p.w_out + ox;
    conv_ubo_out[out_idx] = sum;
}

// ----------------------------------------------------------------------------
// Pass 3: U-Net Skip-Connection Feature-Map Stacking (`stack_skip_features`)
// Stacks decoder and encoder skip feature maps into a contiguous `Texture2DArray`
// (`texture_storage_2d_array<rgba32float, write>`, 4 channels per layer) so the
// GPU indexes directly into concatenation depth without redundant copies.
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> stack_params: SkipStackParams;
@group(0) @binding(1) var<storage, read> stack_dec_in: array<f32>;
@group(0) @binding(2) var<storage, read> stack_skip_in: array<f32>;
@group(0) @binding(3) var stack_tex_out: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn stack_skip_features(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sp = stack_params;
    let x = gid.x;
    let y = gid.y;
    let layer = gid.z;
    if (x >= sp.w || y >= sp.h || layer >= sp.rgba_layers) {
        return;
    }

    let plane = sp.h * sp.w;
    let spatial_idx = y * sp.w + x;
    let base_c = layer * 4u;
    let skip_first = sp.skip_first != 0u;

    var packed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var sub: u32 = 0u; sub < 4u; sub = sub + 1u) {
        let c_concat = base_c + sub;
        if (c_concat < sp.total_c) {
            let is_second_half = c_concat >= sp.c_in;
            let local_c = select(c_concat, c_concat - sp.c_in, is_second_half);
            let src_idx = local_c * plane + spatial_idx;
            let take_skip = select(is_second_half, !is_second_half, skip_first);
            let raw_val = select(stack_dec_in[src_idx], stack_skip_in[src_idx], take_skip);
            packed[sub] = apply_pre_act(raw_val, sp.flags);
        }
    }

    textureStore(stack_tex_out, vec2<i32>(i32(x), i32(y)), i32(layer), packed);
}

// ----------------------------------------------------------------------------
// Pass 4A: Decoder Upsampling ConvTranspose2d (16x16 Block Tiling, Storage Buffers)
// Sub-Pixel Parity Decomposition + hardware `fast_tanh()` and `clamp()` built-ins.
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> deconv_params: LayerParams;
@group(0) @binding(1) var<storage, read> deconv_in_a: array<f32>;
@group(0) @binding(2) var<storage, read> deconv_in_b: array<f32>;
@group(0) @binding(3) var<storage, read> deconv_weights: array<f32>;
@group(0) @binding(4) var<storage, read> deconv_bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> deconv_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn conv_transpose2d_up(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let p = deconv_params;
    var ox: u32 = 0u;
    var oy: u32 = 0u;
    var oc: u32 = 0u;
    var idx: u32 = 0u;

    if (nwg.z == p.c_out && (nwg.y > 1u || p.h_out <= 16u)) {
        ox = gid.x;
        oy = gid.y;
        oc = gid.z;
        if (ox >= p.w_out || oy >= p.h_out || oc >= p.c_out) {
            return;
        }
        idx = (oc * p.h_out + oy) * p.w_out + ox;
    } else {
        idx = wgid.x * 256u + lid.y * 16u + lid.x;
        let total_out = p.c_out * p.h_out * p.w_out;
        if (idx >= total_out) {
            return;
        }
        ox = idx % p.w_out;
        let tmp = idx / p.w_out;
        oy = tmp % p.h_out;
        oc = tmp / p.h_out;
    }

    var sum: f32 = deconv_bias[oc];

    let py = 1u - (oy & 1u);
    let px = 1u - (ox & 1u);
    let base_iy = i32((oy + 1u) >> 1u);
    let base_ix = i32((ox + 1u) >> 1u);

    let has_skip = p.has_skip != 0u;
    let skip_first = has_skip && ((p.flags & 4u) != 0u);
    let total_c_in = select(p.c_in, p.c_in * 2u, has_skip);
    let in_plane = p.h_in * p.w_in;
    let w_oc_base = oc * (total_c_in * 16u);

    // Active-window sub-pixel bounds: eliminates inner-loop branch divergence and
    // reduces bottleneck d1 (1x1 -> 2x2) from 4 parity taps to the 1 active tap per pixel
    let jy_start = u32(max(0, base_iy - i32(p.h_in) + 1));
    let jy_end = u32(clamp(base_iy + 1, 0, 2));
    let jx_start = u32(max(0, base_ix - i32(p.w_in) + 1));
    let jx_end = u32(clamp(base_ix + 1, 0, 2));

    for (var ic: u32 = 0u; ic < p.c_in; ic = ic + 1u) {
        let in_c_base = ic * in_plane;
        let a_chan = select(ic, ic + p.c_in, skip_first);
        let b_chan = select(ic + p.c_in, ic, skip_first);
        let w_a_base = w_oc_base + a_chan * 16u;
        let w_b_base = w_oc_base + b_chan * 16u;

        for (var jy: u32 = jy_start; jy < jy_end; jy = jy + 1u) {
            let iy = u32(base_iy - i32(jy));
            let ky = py + jy * 2u;
            let in_row_base = in_c_base + iy * p.w_in;

            for (var jx: u32 = jx_start; jx < jx_end; jx = jx + 1u) {
                let ix = u32(base_ix - i32(jx));
                let kx = px + jx * 2u;
                let k_offset = ky * 4u + kx;
                let spatial_idx = in_row_base + ix;

                let a = apply_pre_act(deconv_in_a[spatial_idx], p.flags);
                sum = sum + a * deconv_weights[w_a_base + k_offset];

                if (has_skip) {
                    let b = apply_pre_act(deconv_in_b[spatial_idx], p.flags);
                    sum = sum + b * deconv_weights[w_b_base + k_offset];
                }
            }
        }
    }

    if ((p.flags & 8u) != 0u) {
        sum = fast_tanh(sum);
    }

    deconv_out[idx] = sum;
}

// ----------------------------------------------------------------------------
// Pass 4B: Dual-Storage Skip + UBO Weights ConvTranspose2d (`conv_transpose2d_up_ubo`)
// Used when `useUboWeights == true` and `useTexture2DArraySkip == false`.
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> deconv_ubo_params: LayerParams;
@group(0) @binding(1) var<storage, read> deconv_ubo_in_a: array<f32>;
@group(0) @binding(2) var<storage, read> deconv_ubo_in_b: array<f32>;
@group(0) @binding(3) var<uniform> deconv_ubo_only_weights: DeconvUboWeightsD8;
@group(0) @binding(4) var<storage, read_write> deconv_ubo_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn conv_transpose2d_up_ubo(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = deconv_ubo_params;
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;
    if (ox >= p.w_out || oy >= p.h_out || oc >= p.c_out) {
        return;
    }

    var sum: f32 = deconv_ubo_only_weights.bias[oc];

    let py = 1u - (oy & 1u);
    let px = 1u - (ox & 1u);
    let base_iy = i32((oy + 1u) >> 1u);
    let base_ix = i32((ox + 1u) >> 1u);

    let has_skip = p.has_skip != 0u;
    let skip_first = has_skip && ((p.flags & 4u) != 0u);
    let total_c_in = select(p.c_in, p.c_in * 2u, has_skip);
    let in_plane = p.h_in * p.w_in;

    let jy_start = u32(max(0, base_iy - i32(p.h_in) + 1));
    let jy_end = u32(clamp(base_iy + 1, 0, 2));
    let jx_start = u32(max(0, base_ix - i32(p.w_in) + 1));
    let jx_end = u32(clamp(base_ix + 1, 0, 2));

    for (var ic: u32 = 0u; ic < p.c_in; ic = ic + 1u) {
        let in_c_base = ic * in_plane;
        let a_chan = select(ic, ic + p.c_in, skip_first);
        let b_chan = select(ic + p.c_in, ic, skip_first);
        let w_a_vec_base = (oc * total_c_in + a_chan) * 4u;
        let w_b_vec_base = (oc * total_c_in + b_chan) * 4u;

        for (var jy: u32 = jy_start; jy < jy_end; jy = jy + 1u) {
            let iy = u32(base_iy - i32(jy));
            let ky = py + jy * 2u;
            let in_row_base = in_c_base + iy * p.w_in;
            let w_a_row = deconv_ubo_only_weights.weights[w_a_vec_base + ky];
            let w_b_row = deconv_ubo_only_weights.weights[w_b_vec_base + ky];

            for (var jx: u32 = jx_start; jx < jx_end; jx = jx + 1u) {
                let ix = u32(base_ix - i32(jx));
                let kx = px + jx * 2u;
                let spatial_idx = in_row_base + ix;

                let a = apply_pre_act(deconv_ubo_in_a[spatial_idx], p.flags);
                sum = sum + a * w_a_row[kx];

                if (has_skip) {
                    let b = apply_pre_act(deconv_ubo_in_b[spatial_idx], p.flags);
                    sum = sum + b * w_b_row[kx];
                }
            }
        }
    }

    if ((p.flags & 8u) != 0u) {
        sum = fast_tanh(sum);
    }

    let out_idx = (oc * p.h_out + oy) * p.w_out + ox;
    deconv_ubo_out[out_idx] = sum;
}

// ----------------------------------------------------------------------------
// Pass 4C: `Texture2DArray` + UBO Decoder ConvTranspose2d (`conv_transpose2d_up_texarray`)
// Indexes directly into the stacked `texture_2d_array<f32>` skip concatenation depth
// and UBO weights (`DeconvUboWeightsD8`) using 16x16 workgroup tiles.
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> deconv_tex_params: LayerParams;
@group(0) @binding(1) var stacked_skip_tex: texture_2d_array<f32>;
@group(0) @binding(2) var<uniform> deconv_ubo_weights: DeconvUboWeightsD8;
@group(0) @binding(3) var<storage, read_write> deconv_tex_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn conv_transpose2d_up_texarray(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = deconv_tex_params;
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;
    if (ox >= p.w_out || oy >= p.h_out || oc >= p.c_out) {
        return;
    }

    var sum: f32 = deconv_ubo_weights.bias[oc];

    let py = 1u - (oy & 1u);
    let px = 1u - (ox & 1u);
    let base_iy = i32((oy + 1u) >> 1u);
    let base_ix = i32((ox + 1u) >> 1u);

    let jy_start = u32(max(0, base_iy - i32(p.h_in) + 1));
    let jy_end = u32(clamp(base_iy + 1, 0, 2));
    let jx_start = u32(max(0, base_ix - i32(p.w_in) + 1));
    let jx_end = u32(clamp(base_ix + 1, 0, 2));

    let total_c_in = select(p.c_in, p.c_in * 2u, p.has_skip != 0u);
    let rgba_layers = total_c_in >> 2u;

    for (var layer: u32 = 0u; layer < rgba_layers; layer = layer + 1u) {
        let base_c = layer * 4u;

        for (var jy: u32 = jy_start; jy < jy_end; jy = jy + 1u) {
            let iy = base_iy - i32(jy);
            let ky = py + jy * 2u;

            for (var jx: u32 = jx_start; jx < jx_end; jx = jx + 1u) {
                let ix = base_ix - i32(jx);
                let kx = px + jx * 2u;
                let feat4 = textureLoad(stacked_skip_tex, vec2<i32>(ix, iy), i32(layer), 0);

                for (var sub: u32 = 0u; sub < 4u; sub = sub + 1u) {
                    let c_concat = base_c + sub;
                    let w_vec_idx = (oc * total_c_in + c_concat) * 4u + ky;
                    let w_row = deconv_ubo_weights.weights[w_vec_idx];
                    sum = sum + feat4[sub] * w_row[kx];
                }
            }
        }
    }

    if ((p.flags & 8u) != 0u) {
        sum = fast_tanh(sum);
    }

    let out_idx = (oc * p.h_out + oy) * p.w_out + ox;
    deconv_tex_out[out_idx] = sum;
}

// ----------------------------------------------------------------------------
// Pass 4D: `Texture2DArray` + Storage Weights ConvTranspose2d (`conv_transpose2d_up_texarray_storage`)
// Used when `useTexture2DArraySkip == true` and `useUboWeights == false` (or for full
// 16-layer U-Net stages d2..d8 whose weight tensors exceed 64 KiB UBO limits).
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> deconv_tex_storage_params: LayerParams;
@group(0) @binding(1) var stacked_skip_tex_storage: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read> deconv_tex_storage_weights: array<f32>;
@group(0) @binding(3) var<storage, read> deconv_tex_storage_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> deconv_tex_storage_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn conv_transpose2d_up_texarray_storage(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let p = deconv_tex_storage_params;
    var ox: u32 = 0u;
    var oy: u32 = 0u;
    var oc: u32 = 0u;
    var idx: u32 = 0u;

    if (nwg.z == p.c_out && (nwg.y > 1u || p.h_out <= 16u)) {
        ox = gid.x;
        oy = gid.y;
        oc = gid.z;
        if (ox >= p.w_out || oy >= p.h_out || oc >= p.c_out) {
            return;
        }
        idx = (oc * p.h_out + oy) * p.w_out + ox;
    } else {
        idx = wgid.x * 256u + lid.y * 16u + lid.x;
        let total_out = p.c_out * p.h_out * p.w_out;
        if (idx >= total_out) {
            return;
        }
        ox = idx % p.w_out;
        let tmp = idx / p.w_out;
        oy = tmp % p.h_out;
        oc = tmp / p.h_out;
    }

    var sum: f32 = deconv_tex_storage_bias[oc];

    let py = 1u - (oy & 1u);
    let px = 1u - (ox & 1u);
    let base_iy = i32((oy + 1u) >> 1u);
    let base_ix = i32((ox + 1u) >> 1u);

    let jy_start = u32(max(0, base_iy - i32(p.h_in) + 1));
    let jy_end = u32(clamp(base_iy + 1, 0, 2));
    let jx_start = u32(max(0, base_ix - i32(p.w_in) + 1));
    let jx_end = u32(clamp(base_ix + 1, 0, 2));

    let total_c_in = select(p.c_in, p.c_in * 2u, p.has_skip != 0u);
    let rgba_layers = total_c_in >> 2u;
    let w_oc_base = oc * (total_c_in * 16u);

    for (var layer: u32 = 0u; layer < rgba_layers; layer = layer + 1u) {
        let base_c = layer * 4u;

        for (var jy: u32 = jy_start; jy < jy_end; jy = jy + 1u) {
            let iy = base_iy - i32(jy);
            let ky = py + jy * 2u;

            for (var jx: u32 = jx_start; jx < jx_end; jx = jx + 1u) {
                let ix = base_ix - i32(jx);
                let kx = px + jx * 2u;
                let k_offset = ky * 4u + kx;
                let feat4 = textureLoad(stacked_skip_tex_storage, vec2<i32>(ix, iy), i32(layer), 0);

                for (var sub: u32 = 0u; sub < 4u; sub = sub + 1u) {
                    let c_concat = base_c + sub;
                    let w_idx = w_oc_base + c_concat * 16u + k_offset;
                    sum = sum + feat4[sub] * deconv_tex_storage_weights[w_idx];
                }
            }
        }
    }

    if ((p.flags & 8u) != 0u) {
        sum = fast_tanh(sum);
    }

    deconv_tex_storage_out[idx] = sum;
}

// ----------------------------------------------------------------------------
// Pass 5: Fused Instance Normalization + Stochastic PCG Dropout (`16x16` Workgroup)
// Supports both:
//   - `use_welford == 0u`: Two-Pass Centered Variance Reduction (`E[(X - mu)^2]`, 17 barriers)
//   - `use_welford != 0u`: Single-Pass Welford Online + Chan Pairwise Tree Reduction (9 barriers)
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> norm_params: NormParams;
@group(0) @binding(1) var<storage, read> norm_gamma: array<f32>;
@group(0) @binding(2) var<storage, read> norm_beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_data: array<f32>;

var<workgroup> wg_sum: array<f32, 256>;
var<workgroup> wg_m2: array<f32, 256>;
var<workgroup> wg_count: array<f32, 256>;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(16, 16, 1)
fn instance_norm_dropout(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let c = wgid.x;
    let tid = lid.y * 16u + lid.x;
    let n = norm_params.h * norm_params.w;
    let offset = c * n;
    // Adaptive reduction stride: for small bottleneck maps (e7/d1: n=4 -> stride=2; e6/d2: n=16 -> stride=8;
    // e5/d3: n=64 -> stride=32), skips redundant barrier stages over idle threads (tid >= n).
    let initial_stride = min(128u, max(n >> 1u, 1u));
    let gamma = norm_gamma[c];
    let beta = norm_beta[c];

    // BatchNorm eval can be folded at export time into scale/shift. In that
    // mode there is deliberately no spatial reduction: the stored per-channel
    // gamma/beta already encode running_mean/running_var and affine params.
    if (norm_params.mode == 1u) {
        for (var i: u32 = tid; i < n; i = i + 256u) {
            let idx = offset + i;
            var val = norm_data[idx] * gamma + beta;
            if (norm_params.apply_dropout != 0u) {
                let rng = pcg_hash(idx ^ norm_params.seed);
                let keep = select(0.0, 2.0, (rng & 1u) == 1u);
                val = val * keep;
            }
            norm_data[idx] = val;
        }
        return;
    }

    var mean: f32 = 0.0;
    var variance: f32 = 0.0;

    if (norm_params.use_welford != 0u) {
        // Single-Pass Welford Online Accumulation per thread + Chan Pairwise Tree Reduction
        var w_mean: f32 = 0.0;
        var w_m2: f32 = 0.0;
        var w_count: f32 = 0.0;

        for (var i: u32 = tid; i < n; i = i + 256u) {
            let x = norm_data[offset + i];
            w_count = w_count + 1.0;
            let delta = x - w_mean;
            w_mean = w_mean + (delta / w_count);
            let delta2 = x - w_mean;
            w_m2 = w_m2 + delta * delta2;
        }

        wg_sum[tid] = w_mean;
        wg_m2[tid] = w_m2;
        wg_count[tid] = w_count;
        workgroupBarrier();

        for (var stride: u32 = initial_stride; stride > 0u; stride = stride >> 1u) {
            if (tid < stride) {
                let count_a = wg_count[tid];
                let count_b = wg_count[tid + stride];
                let count_ab = count_a + count_b;
                if (count_ab > 0.0) {
                    let delta = wg_sum[tid + stride] - wg_sum[tid];
                    wg_sum[tid] = wg_sum[tid] + delta * (count_b / count_ab);
                    wg_m2[tid] = wg_m2[tid] + wg_m2[tid + stride] + delta * delta * ((count_a * count_b) / count_ab);
                    wg_count[tid] = count_ab;
                }
            }
            workgroupBarrier();
        }

        mean = wg_sum[0];
        variance = clamp(wg_m2[0] / f32(max(n, 1u)), 0.0, 1e12);
    } else {
        // Pass 1: Parallel reduction for channel mean `mu`
        var local_sum: f32 = 0.0;
        for (var i: u32 = tid; i < n; i = i + 256u) {
            local_sum = local_sum + norm_data[offset + i];
        }
        wg_sum[tid] = local_sum;
        workgroupBarrier();

        for (var stride: u32 = initial_stride; stride > 0u; stride = stride >> 1u) {
            if (tid < stride) {
                wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
            }
            workgroupBarrier();
        }

        mean = wg_sum[0] / f32(n);
        workgroupBarrier();

        // Pass 2: Parallel reduction for centered sum of squares `sum((x - mean)^2)`
        var local_var_sum: f32 = 0.0;
        for (var i: u32 = tid; i < n; i = i + 256u) {
            let diff = norm_data[offset + i] - mean;
            local_var_sum = local_var_sum + diff * diff;
        }
        wg_sum[tid] = local_var_sum;
        workgroupBarrier();

        for (var stride: u32 = initial_stride; stride > 0u; stride = stride >> 1u) {
            if (tid < stride) {
                wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
            }
            workgroupBarrier();
        }

        variance = clamp(wg_sum[0] / f32(n), 0.0, 1e12);
    }

    let inv_std = inverseSqrt(variance + norm_params.eps);

    // Normalize, scale/shift, and apply optional inverted 50% dropout
    for (var i: u32 = tid; i < n; i = i + 256u) {
        let idx = offset + i;
        var val = (norm_data[idx] - mean) * inv_std * gamma + beta;
        if (norm_params.apply_dropout != 0u) {
            let rng = pcg_hash(idx ^ norm_params.seed);
            let keep = select(0.0, 2.0, (rng & 1u) == 1u);
            val = val * keep;
        }
        norm_data[idx] = val;
    }
}

// ----------------------------------------------------------------------------
// Pass 6: On-Device Mean Squared Error (MSE) Verification Kernel (`compute_mse`)
// Verifies the entire U-Net forward pass on-device by reducing squared and
// absolute differences between `gpu_final_out` and `cpu_ref_out` (`d8`).
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> mse_params: MseParams;
@group(0) @binding(1) var<storage, read> gpu_final_out: array<f32>;
@group(0) @binding(2) var<storage, read> cpu_ref_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> mse_stats_out: MseStatsOut;

var<workgroup> wg_mse_sum: array<f32, 256>;
var<workgroup> wg_mse_max: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn compute_mse(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let total = mse_params.total_elements;

    var local_sq_err: f32 = 0.0;
    var local_max_err: f32 = 0.0;

    for (var i: u32 = tid; i < total; i = i + 256u) {
        let diff = abs(gpu_final_out[i] - cpu_ref_out[i]);
        local_sq_err = local_sq_err + diff * diff;
        local_max_err = max(local_max_err, diff);
    }

    wg_mse_sum[tid] = local_sq_err;
    wg_mse_max[tid] = local_max_err;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            wg_mse_sum[tid] = wg_mse_sum[tid] + wg_mse_sum[tid + stride];
            wg_mse_max[tid] = max(wg_mse_max[tid], wg_mse_max[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let mse = wg_mse_sum[0] / f32(max(total, 1u));
        let rmse = sqrt(mse);
        let max_err = wg_mse_max[0];
        // Peak signal range is [-1, 1] -> peak_sq = 4.0
        let safe_mse = clamp(mse, 1e-14, 4.0);
        let psnr = select(99.99, 10.0 * log(4.0 / safe_mse) / log(10.0), mse > 1e-12);
        mse_stats_out.mse = mse;
        mse_stats_out.rmse = rmse;
        mse_stats_out.max_abs_err = max_err;
        mse_stats_out.psnr_db = psnr;
    }
}

// ----------------------------------------------------------------------------
// Pass 7: Full-Screen Triangle Render Pass (NCHW [-1, 1] -> Swapchain Texture)
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> final_out: array<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = pos[vi];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let x = clamp(u32(in.uv.x * 256.0), 0u, 255u);
    let y = clamp(u32(in.uv.y * 256.0), 0u, 255u);
    let idx = y * 256u + x;
    let plane = 65536u;
    let r = final_out[0u * plane + idx] * 0.5 + 0.5;
    let g = final_out[1u * plane + idx] * 0.5 + 0.5;
    let b = final_out[2u * plane + idx] * 0.5 + 0.5;
    return vec4<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
}
