use bytemuck::{Pod, Zeroable};
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

/// Expected exact byte size of standardized `pix2pix_unet256.bin`:
/// 54,419,459 float32 parameters = 217,677,836 bytes.
pub const EXPECTED_WEIGHTS_BYTES: usize = 217_677_836;

/// Maximum chunk size (4 MiB) used when streaming weights from a `js_sys::Uint8Array`
/// into WebGPU buffers so Wasm linear memory never allocates a 217.68 MB buffer.
const STREAM_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Pads any byte length up to a 16-byte boundary to satisfy `@align(16)` WebGPU rules.
#[inline]
pub const fn align_to_16_bytes(bytes: usize) -> u64 {
    ((bytes + 15) & !15) as u64
}

#[wasm_bindgen]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DropoutConvention {
    /// Original Torch7 (`phillipi/pix2pix/models.lua`) convention: 50% dropout on d1, d2, d3.
    Torch7D1ToD3 = 0,
    /// PyTorch (`junyanz/pytorch-CycleGAN-and-pix2pix`) convention: 50% dropout on d2, d3, d4.
    PyTorchD2ToD4 = 1,
    /// Deterministic inference with test-time dropout disabled on all layers.
    Disabled = 2,
}

#[wasm_bindgen]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NormalizationConvention {
    /// Compute spatial InstanceNorm statistics from each current activation.
    InstanceNorm = 0,
    /// Apply export-folded BatchNorm eval as per-channel `x * scale + shift`.
    BatchNormEvalFolded = 1,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LayerParams {
    pub h_in: u32,
    pub w_in: u32,
    pub c_in: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub c_out: u32,
    pub flags: u32,
    pub has_skip: u32,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NormParams {
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub apply_dropout: u32,
    pub seed: u32,
    pub eps: f32,
    pub use_welford: u32,
    pub mode: u32,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SkipStackParams {
    pub h: u32,
    pub w: u32,
    pub c_in: u32,
    pub skip_first: u32,
    pub total_c: u32,
    pub rgba_layers: u32,
    pub flags: u32,
    pub _pad0: u32,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MseParams {
    pub total_elements: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

struct ConvStep {
    bind_group: wgpu::BindGroup,
    param_buf: wgpu::Buffer,
    base_params: LayerParams,
    workgroups: (u32, u32, u32),
}

struct SkipStackStep {
    stack_bind_group: wgpu::BindGroup,
    stack_param_buf: wgpu::Buffer,
    base_stack_params: SkipStackParams,
    stack_workgroups: (u32, u32, u32),
    deconv_texarray_bind_group: wgpu::BindGroup,
}

struct DecoderStage {
    deconv: ConvStep,
    skip_stack: Option<SkipStackStep>,
    norm: Option<NormStep>,
}

struct NormStep {
    bind_group: wgpu::BindGroup,
    param_buf: wgpu::Buffer,
    base_params: NormParams,
    workgroups_x: u32,
}

fn compute_16x16_dispatch(h_out: u32, w_out: u32, c_out: u32) -> (u32, u32, u32) {
    if h_out >= 16 && w_out >= 16 {
        (w_out.div_ceil(16), h_out.div_ceil(16), c_out)
    } else {
        ((c_out * h_out * w_out).div_ceil(256), 1, 1)
    }
}

/// Trait abstracting over either a borrowed Rust slice `&[u8]` or a zero-Wasm-heap-spike
/// `js_sys::Uint8Array` reader that copies layer weights in <= 4 MiB chunks directly to the GPU.
trait WeightSource {
    fn create_weight_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        count_floats: usize,
    ) -> Result<wgpu::Buffer, JsValue>;
}

struct SliceWeightReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> SliceWeightReader<'a> {
    fn new(data: &'a [u8]) -> Result<Self, JsValue> {
        if data.len() != EXPECTED_WEIGHTS_BYTES {
            return Err(JsValue::from_str(&format!(
                "Invalid pix2pix_unet256.bin size: expected {} bytes (54,419,459 f32s), got {} bytes",
                EXPECTED_WEIGHTS_BYTES,
                data.len()
            )));
        }
        Ok(Self { data, offset: 0 })
    }
}

impl<'a> WeightSource for SliceWeightReader<'a> {
    fn create_weight_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        count_floats: usize,
    ) -> Result<wgpu::Buffer, JsValue> {
        let bytes = count_floats
            .checked_mul(4)
            .ok_or_else(|| JsValue::from_str("Weight byte count overflow"))?;
        let end = self
            .offset
            .checked_add(bytes)
            .ok_or_else(|| JsValue::from_str("Weight offset overflow"))?;
        if end > self.data.len() {
            return Err(JsValue::from_str(&format!(
                "Weight buffer underrun at byte offset {}: requested {} bytes, remaining {} bytes",
                self.offset,
                bytes,
                self.data.len().saturating_sub(self.offset)
            )));
        }
        let slice = &self.data[self.offset..end];
        self.offset = end;

        let aligned_size = align_to_16_bytes(bytes);
        if aligned_size == bytes as u64 {
            Ok(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: slice,
                usage: wgpu::BufferUsages::STORAGE,
            }))
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: aligned_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, slice);
            Ok(buf)
        }
    }
}

/// Zero-Wasm-heap-spike reader backed by `js_sys::Uint8Array`.
/// Copies at most `STREAM_CHUNK_BYTES` (4 MiB) at a time from JS into Wasm linear memory
/// and uploads via `queue.write_buffer`, avoiding the 217.68 MB Wasm heap allocation.
struct JsUint8WeightReader<'a> {
    array: &'a js_sys::Uint8Array,
    total_len: usize,
    offset: usize,
    scratch: Vec<u8>,
}

impl<'a> JsUint8WeightReader<'a> {
    fn new(array: &'a js_sys::Uint8Array) -> Result<Self, JsValue> {
        let total_len = array.byte_length() as usize;
        if total_len != EXPECTED_WEIGHTS_BYTES {
            return Err(JsValue::from_str(&format!(
                "Invalid pix2pix_unet256.bin Uint8Array size: expected {} bytes, got {} bytes",
                EXPECTED_WEIGHTS_BYTES, total_len
            )));
        }
        Ok(Self {
            array,
            total_len,
            offset: 0,
            scratch: vec![0u8; STREAM_CHUNK_BYTES],
        })
    }
}

impl<'a> WeightSource for JsUint8WeightReader<'a> {
    fn create_weight_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        count_floats: usize,
    ) -> Result<wgpu::Buffer, JsValue> {
        let bytes = count_floats
            .checked_mul(4)
            .ok_or_else(|| JsValue::from_str("Weight byte count overflow"))?;
        let end = self
            .offset
            .checked_add(bytes)
            .ok_or_else(|| JsValue::from_str("Weight offset overflow"))?;
        if end > self.total_len {
            return Err(JsValue::from_str(&format!(
                "Uint8Array weight underrun at byte offset {}: requested {} bytes, remaining {} bytes",
                self.offset,
                bytes,
                self.total_len.saturating_sub(self.offset)
            )));
        }

        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: align_to_16_bytes(bytes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut written = 0usize;
        while written < bytes {
            let chunk_len = (bytes - written).min(STREAM_CHUNK_BYTES);
            let sub_start = (self.offset + written) as u32;
            let sub_end = sub_start + (chunk_len as u32);
            let view = self.array.subarray(sub_start, sub_end);
            view.copy_to(&mut self.scratch[..chunk_len]);
            queue.write_buffer(&buf, written as u64, &self.scratch[..chunk_len]);
            written += chunk_len;
        }

        self.offset = end;
        Ok(buf)
    }
}

#[wasm_bindgen]
pub struct Pix2PixEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    input_texture: wgpu::Texture,
    preprocess_pipeline: wgpu::ComputePipeline,
    preprocess_bg: wgpu::BindGroup,
    conv_pipeline: wgpu::ComputePipeline,
    stack_skip_pipeline: wgpu::ComputePipeline,
    deconv_pipeline: wgpu::ComputePipeline,
    deconv_texarray_storage_pipeline: wgpu::ComputePipeline,
    norm_pipeline: wgpu::ComputePipeline,
    mse_pipeline: wgpu::ComputePipeline,
    mse_bg: wgpu::BindGroup,
    cpu_ref_buf: wgpu::Buffer,
    mse_stats_gpu_buf: wgpu::Buffer,
    mse_staging_read_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_staging_read_buf: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    render_bg: wgpu::BindGroup,
    encoders: Vec<(ConvStep, Option<NormStep>)>,
    decoders: Vec<DecoderStage>,
    dropout_convention: DropoutConvention,
    normalization_convention: NormalizationConvention,
    use_texture2d_array_skip: bool,
    frame_counter: u32,
}

#[wasm_bindgen]
impl Pix2PixEngine {
    /// Creates the WebGPU Pix2Pix U-Net 256 engine from a borrowed byte slice.
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        weights_bin: &[u8],
    ) -> Result<Pix2PixEngine, JsValue> {
        console_error_panic_hook::set_once();
        let mut reader = SliceWeightReader::new(weights_bin)?;
        Self::build_with_source(canvas, &mut reader, DropoutConvention::Torch7D1ToD3).await
    }

    /// Creates the WebGPU Pix2Pix U-Net 256 engine by streaming 4 MiB chunks directly from
    /// a `js_sys::Uint8Array`, eliminating the 217.68 MB Wasm linear memory allocation spike.
    pub async fn create_from_uint8_array(
        canvas: web_sys::HtmlCanvasElement,
        weights_array: js_sys::Uint8Array,
        dropout_convention: DropoutConvention,
    ) -> Result<Pix2PixEngine, JsValue> {
        console_error_panic_hook::set_once();
        let mut reader = JsUint8WeightReader::new(&weights_array)?;
        Self::build_with_source(canvas, &mut reader, dropout_convention).await
    }

    async fn build_with_source<S: WeightSource>(
        canvas: web_sys::HtmlCanvasElement,
        reader: &mut S,
        dropout_convention: DropoutConvention,
    ) -> Result<Pix2PixEngine, JsValue> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Surface creation error: {e:?}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| JsValue::from_str("No compatible WebGPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("pix2pix_unet256_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Request device error: {e:?}")))?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .first()
            .copied()
            .ok_or_else(|| JsValue::from_str("Surface exposes no supported texture formats"))?;

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: 256,
                height: 256,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pix2pix_wgsl_shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        // 1. Input RGBA8 Texture + Preprocessing Compute Pipeline
        let input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("input_sketch_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let alloc_tensor = |label: &str, floats: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: align_to_16_bytes(floats * 4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let in_buf = alloc_tensor("in_buf_3x256x256", 3 * 256 * 256);

        let preprocess_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("preprocess_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("preprocess"),
                compilation_options: Default::default(),
                cache: None,
            });

        let preprocess_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preprocess_bg"),
            layout: &preprocess_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
            ],
        });

        let conv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv2d_down_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("conv2d_down"),
            compilation_options: Default::default(),
            cache: None,
        });

        let stack_skip_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("stack_skip_features_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("stack_skip_features"),
                compilation_options: Default::default(),
                cache: None,
            });

        let deconv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_transpose2d_up_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("conv_transpose2d_up"),
            compilation_options: Default::default(),
            cache: None,
        });

        let deconv_texarray_storage_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("conv_transpose2d_up_texarray_storage_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("conv_transpose2d_up_texarray_storage"),
                compilation_options: Default::default(),
                cache: None,
            });

        let norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_norm_dropout_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("instance_norm_dropout"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_mse_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("compute_mse"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 2. Encoder Topology (e1..e8)
        let enc_specs: [(u32, u32, u32, u32, u32, bool); 8] = [
            (256, 3, 128, 64, 0, false),    // e1: no pre-act, no norm
            (128, 64, 64, 128, 1, true),    // e2: pre-LeakyReLU, norm
            (64, 128, 32, 256, 1, true),    // e3: pre-LeakyReLU, norm
            (32, 256, 16, 512, 1, true),    // e4: pre-LeakyReLU, norm
            (16, 512, 8, 512, 1, true),     // e5: pre-LeakyReLU, norm
            (8, 512, 4, 512, 1, true),      // e6: pre-LeakyReLU, norm
            (4, 512, 2, 512, 1, true),      // e7: pre-LeakyReLU, norm
            (2, 512, 1, 512, 1, false),     // e8: pre-LeakyReLU, bottleneck (no norm)
        ];

        let mut e_bufs = Vec::with_capacity(8);
        let mut encoders = Vec::with_capacity(8);

        for (i, &(h_in, c_in, h_out, c_out, flags, has_norm)) in enc_specs.iter().enumerate() {
            let out_buf = alloc_tensor(
                &format!("e{}_buf", i + 1),
                (c_out * h_out * h_out) as usize,
            );
            let prev_buf = if i == 0 { &in_buf } else { &e_bufs[i - 1] };

            let w_buf = reader.create_weight_buffer(
                &device,
                &queue,
                &format!("e{}_weights", i + 1),
                (c_out * c_in * 16) as usize,
            )?;
            let b_buf = reader.create_weight_buffer(
                &device,
                &queue,
                &format!("e{}_bias", i + 1),
                c_out as usize,
            )?;

            let base_params = LayerParams {
                h_in,
                w_in: h_in,
                c_in,
                h_out,
                w_out: h_out,
                c_out,
                flags,
                has_skip: 0,
            };
            let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("e{}_params", i + 1)),
                contents: bytemuck::bytes_of(&base_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("e{}_bg", i + 1)),
                layout: &conv_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prev_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });

            let conv_step = ConvStep {
                bind_group: bg,
                param_buf,
                base_params,
                workgroups: compute_16x16_dispatch(h_out, h_out, c_out),
            };

            let norm_step = if has_norm {
                let g_buf = reader.create_weight_buffer(
                    &device,
                    &queue,
                    &format!("e{}_norm_gamma", i + 1),
                    c_out as usize,
                )?;
                let beta_buf = reader.create_weight_buffer(
                    &device,
                    &queue,
                    &format!("e{}_norm_beta", i + 1),
                    c_out as usize,
                )?;

                let base_norm_params = NormParams {
                    h: h_out,
                    w: h_out,
                    c: c_out,
                    apply_dropout: 0,
                    seed: 0,
                    eps: 1e-5,
                    use_welford: 0,
                    mode: NormalizationConvention::InstanceNorm as u32,
                };
                let np_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("e{}_norm_params", i + 1)),
                    contents: bytemuck::bytes_of(&base_norm_params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let n_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("e{}_norm_bg", i + 1)),
                    layout: &norm_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: np_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: g_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: beta_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: out_buf.as_entire_binding(),
                        },
                    ],
                });

                Some(NormStep {
                    bind_group: n_bg,
                    param_buf: np_buf,
                    base_params: base_norm_params,
                    workgroups_x: c_out,
                })
            } else {
                None
            };

            e_bufs.push(out_buf);
            encoders.push((conv_step, norm_step));
        }

        // 3. Decoder Topology (d1..d8)
        let dec_specs: [(u32, u32, u32, u32, u32, bool, bool); 8] = [
            // (h_in, c_in_single, h_out, c_out, flags, has_skip, has_norm)
            (1, 512, 2, 512, 2, false, true),    // d1: input e8
            (2, 512, 4, 512, 2, true, true),     // d2: input d1 + e7
            (4, 512, 8, 512, 2, true, true),     // d3: input d2 + e6
            (8, 512, 16, 512, 2, true, true),    // d4: input d3 + e5
            (16, 512, 32, 256, 2, true, true),   // d5: input d4 + e4
            (32, 256, 64, 128, 2, true, true),   // d6: input d5 + e3
            (64, 128, 128, 64, 2, true, true),   // d7: input d6 + e2
            (128, 64, 256, 3, 10, true, false),  // d8: input d7 + e1, pre-ReLU(2) + Tanh(8) = 10
        ];

        // 16-byte aligned zero-initialized buffer for d1's unused second input binding (`deconv_in_b`)
        let empty_skip_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("d1_empty_skip_buf_16b"),
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut d_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(8);
        let mut decoders = Vec::with_capacity(8);

        for (i, &(h_in, c_in, h_out, c_out, flags, has_skip, has_norm)) in
            dec_specs.iter().enumerate()
        {
            let dropout = Self::should_apply_dropout(dropout_convention, i);
            let out_buf = alloc_tensor(
                &format!("d{}_buf", i + 1),
                (c_out * h_out * h_out) as usize,
            );
            let in_a: &wgpu::Buffer = if i == 0 { &e_bufs[7] } else { &d_bufs[i - 1] };
            let in_b: &wgpu::Buffer = if has_skip {
                &e_bufs[7 - i]
            } else {
                &empty_skip_buf
            };

            let total_c_in = if has_skip { c_in * 2 } else { c_in };
            let w_buf = reader.create_weight_buffer(
                &device,
                &queue,
                &format!("d{}_weights", i + 1),
                (c_out * total_c_in * 16) as usize,
            )?;
            let b_buf = reader.create_weight_buffer(
                &device,
                &queue,
                &format!("d{}_bias", i + 1),
                c_out as usize,
            )?;

            let base_params = LayerParams {
                h_in,
                w_in: h_in,
                c_in,
                h_out,
                w_out: h_out,
                c_out,
                flags,
                has_skip: u32::from(has_skip),
            };
            let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("d{}_params", i + 1)),
                contents: bytemuck::bytes_of(&base_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("d{}_bg", i + 1)),
                layout: &deconv_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: in_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: in_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });

            let skip_stack_step = if has_skip {
                let rgba_layers = total_c_in / 4;
                let skip_tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("d{}_skip_tex2d_array", i + 1)),
                    size: wgpu::Extent3d {
                        width: h_in,
                        height: h_in,
                        depth_or_array_layers: rgba_layers,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let skip_tex_view = skip_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("d{}_skip_tex2d_array_view", i + 1)),
                    format: Some(wgpu::TextureFormat::Rgba32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(rgba_layers),
                });

                let base_stack_params = SkipStackParams {
                    h: h_in,
                    w: h_in,
                    c_in,
                    skip_first: 0,
                    total_c: total_c_in,
                    rgba_layers,
                    flags: flags & 3, // Pre-activation bits (ReLU = 2)
                    _pad0: 0,
                };
                let stack_param_buf =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("d{}_stack_params", i + 1)),
                        contents: bytemuck::bytes_of(&base_stack_params),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                let stack_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("d{}_stack_bg", i + 1)),
                    layout: &stack_skip_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: stack_param_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: in_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: in_b.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&skip_tex_view),
                        },
                    ],
                });

                let deconv_texarray_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("d{}_deconv_texarray_bg", i + 1)),
                    layout: &deconv_texarray_storage_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&skip_tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: w_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buf.as_entire_binding(),
                        },
                    ],
                });

                Some(SkipStackStep {
                    stack_bind_group: stack_bg,
                    stack_param_buf,
                    base_stack_params,
                    stack_workgroups: (h_in.div_ceil(16), h_in.div_ceil(16), rgba_layers),
                    deconv_texarray_bind_group: deconv_texarray_bg,
                })
            } else {
                None
            };

            let deconv_step = ConvStep {
                bind_group: bg,
                param_buf,
                base_params,
                workgroups: compute_16x16_dispatch(h_out, h_out, c_out),
            };

            let norm_step = if has_norm {
                let g_buf = reader.create_weight_buffer(
                    &device,
                    &queue,
                    &format!("d{}_norm_gamma", i + 1),
                    c_out as usize,
                )?;
                let beta_buf = reader.create_weight_buffer(
                    &device,
                    &queue,
                    &format!("d{}_norm_beta", i + 1),
                    c_out as usize,
                )?;

                let base_norm_params = NormParams {
                    h: h_out,
                    w: h_out,
                    c: c_out,
                    apply_dropout: u32::from(dropout),
                    seed: (1337 + i as u32) * 9973,
                    eps: 1e-5,
                    use_welford: 0,
                    mode: NormalizationConvention::InstanceNorm as u32,
                };
                let np_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("d{}_norm_params", i + 1)),
                    contents: bytemuck::bytes_of(&base_norm_params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let n_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("d{}_norm_bg", i + 1)),
                    layout: &norm_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: np_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: g_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: beta_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: out_buf.as_entire_binding(),
                        },
                    ],
                });

                Some(NormStep {
                    bind_group: n_bg,
                    param_buf: np_buf,
                    base_params: base_norm_params,
                    workgroups_x: c_out,
                })
            } else {
                None
            };

            d_bufs.push(out_buf);
            decoders.push(DecoderStage {
                deconv: deconv_step,
                skip_stack: skip_stack_step,
                norm: norm_step,
            });
        }

        let output_buf = d_bufs
            .pop()
            .ok_or_else(|| JsValue::from_str("decoder topology produced no final output buffer"))?;
        let output_staging_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("d8_output_staging_read_buf"),
            size: (3 * 256 * 256 * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // 4. On-Device MSE Verification Buffers & Bind Group (`compute_mse` on d8 [3, 256, 256])
        let mse_params = MseParams {
            total_elements: 3 * 256 * 256,
            width: 256,
            height: 256,
            channels: 3,
        };
        let mse_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("d8_mse_params_ubo"),
            contents: bytemuck::bytes_of(&mse_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cpu_ref_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("d8_cpu_ref_tensor_buf"),
            size: align_to_16_bytes(3 * 256 * 256 * 4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mse_stats_gpu_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("d8_mse_stats_gpu_buf"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mse_staging_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("d8_mse_staging_read_buf"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mse_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("d8_mse_bg"),
            layout: &mse_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mse_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cpu_ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mse_stats_gpu_buf.as_entire_binding(),
                },
            ],
        });

        // 5. Full-Screen Triangle Render Pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buf.as_entire_binding(),
            }],
        });

        Ok(Pix2PixEngine {
            device,
            queue,
            surface,
            input_texture,
            preprocess_pipeline,
            preprocess_bg,
            conv_pipeline,
            stack_skip_pipeline,
            deconv_pipeline,
            deconv_texarray_storage_pipeline,
            norm_pipeline,
            mse_pipeline,
            mse_bg,
            cpu_ref_buf,
            mse_stats_gpu_buf,
            mse_staging_read_buf,
            output_buf,
            output_staging_read_buf,
            render_pipeline,
            render_bg,
            encoders,
            decoders,
            dropout_convention,
            normalization_convention: NormalizationConvention::InstanceNorm,
            use_texture2d_array_skip: false,
            frame_counter: 0,
        })
    }

    fn should_apply_dropout(convention: DropoutConvention, dec_idx: usize) -> bool {
        match convention {
            DropoutConvention::Torch7D1ToD3 => matches!(dec_idx, 0..=2), // d1, d2, d3
            DropoutConvention::PyTorchD2ToD4 => matches!(dec_idx, 1..=3), // d2, d3, d4
            DropoutConvention::Disabled => false,
        }
    }

    /// Dynamically switches the active dropout layer placement convention (`d1..d3`, `d2..d4`, or `Disabled`).
    pub fn set_dropout_convention(&mut self, convention: DropoutConvention) {
        self.dropout_convention = convention;
        for (idx, stage) in self.decoders.iter_mut().enumerate() {
            if let Some(norm) = &mut stage.norm {
                norm.base_params.apply_dropout =
                    u32::from(Self::should_apply_dropout(convention, idx));
                self.queue
                    .write_buffer(&norm.param_buf, 0, bytemuck::bytes_of(&norm.base_params));
            }
        }
    }

    /// Switches normalization semantics across all normalized encoder/decoder stages.
    ///
    /// `BatchNormEvalFolded` requires weights exported with running-statistics
    /// folded into the stored per-channel gamma/beta slots.
    pub fn set_normalization_convention(&mut self, convention: NormalizationConvention) {
        self.normalization_convention = convention;
        let mode = convention as u32;
        for (_, norm_opt) in &mut self.encoders {
            if let Some(norm) = norm_opt {
                norm.base_params.mode = mode;
                self.queue
                    .write_buffer(&norm.param_buf, 0, bytemuck::bytes_of(&norm.base_params));
            }
        }
        for stage in &mut self.decoders {
            if let Some(norm) = &mut stage.norm {
                norm.base_params.mode = mode;
                self.queue
                    .write_buffer(&norm.param_buf, 0, bytemuck::bytes_of(&norm.base_params));
            }
        }
    }

    /// Toggles hardware-level `[skip, decoder]` vs `[decoder, skip]` channel order bit (bit 2 = 4u)
    /// across all skip-connected decoder layers (`d2..d8`).
    pub fn set_raw_pytorch_skip_order(&mut self, raw_skip_first: bool) {
        for stage in &mut self.decoders {
            if stage.deconv.base_params.has_skip != 0 {
                if raw_skip_first {
                    stage.deconv.base_params.flags |= 4u32;
                } else {
                    stage.deconv.base_params.flags &= !4u32;
                }
                self.queue.write_buffer(
                    &stage.deconv.param_buf,
                    0,
                    bytemuck::bytes_of(&stage.deconv.base_params),
                );
            }
            if let Some(skip_stack) = &mut stage.skip_stack {
                skip_stack.base_stack_params.skip_first = u32::from(raw_skip_first);
                self.queue.write_buffer(
                    &skip_stack.stack_param_buf,
                    0,
                    bytemuck::bytes_of(&skip_stack.base_stack_params),
                );
            }
        }
    }

    /// Toggles between 2-Pass Centered FP32 (`false`) and 1-Pass Welford + Chan Tree Reduction (`true`)
    /// across all encoder and decoder InstanceNorm layers.
    pub fn set_use_welford(&mut self, use_welford: bool) {
        let flag = u32::from(use_welford);
        for (_, norm_opt) in &mut self.encoders {
            if let Some(norm) = norm_opt {
                norm.base_params.use_welford = flag;
                self.queue
                    .write_buffer(&norm.param_buf, 0, bytemuck::bytes_of(&norm.base_params));
            }
        }
        for stage in &mut self.decoders {
            if let Some(norm) = &mut stage.norm {
                norm.base_params.use_welford = flag;
                self.queue
                    .write_buffer(&norm.param_buf, 0, bytemuck::bytes_of(&norm.base_params));
            }
        }
    }

    /// Toggles between dual-storage-buffer skip concatenation (`false`) and contiguous
    /// `Texture2DArray` (`rgba32float`) skip-feature stacking (`true`) on `d2..d8`.
    pub fn set_use_texture2d_array_skip(&mut self, enabled: bool) {
        self.use_texture2d_array_skip = enabled;
    }

    /// Executes the on-device WGSL `compute_mse` verification kernel comparing the GPU's
    /// most recent `d8` output tensor (`[3, 256, 256]`) against a CPU reference tensor,
    /// returning `[mse, rmse, max_abs_err, psnr_db]`.
    pub async fn compute_on_device_mse(&mut self, cpu_ref_nchw: &[f32]) -> Result<Vec<f32>, JsValue> {
        if cpu_ref_nchw.len() != 3 * 256 * 256 {
            return Err(JsValue::from_str(&format!(
                "Expected 196,608 float32 elements (3x256x256) for cpu_ref_nchw, got {}",
                cpu_ref_nchw.len()
            )));
        }

        self.queue
            .write_buffer(&self.cpu_ref_buf, 0, bytemuck::cast_slice(cpu_ref_nchw));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pix2pix_on_device_mse_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_mse_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.mse_pipeline);
            cpass.set_bind_group(0, &self.mse_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.mse_stats_gpu_buf, 0, &self.mse_staging_read_buf, 0, 16);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.mse_staging_read_buf.slice(..);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| match res {
                Ok(()) => {
                    let _ = resolve.call0(&JsValue::NULL);
                }
                Err(e) => {
                    let _ = reject.call1(&JsValue::NULL, &JsValue::from_str(&format!("{e:?}")));
                }
            });
        });

        let _ = self.device.poll(wgpu::Maintain::Poll);
        wasm_bindgen_futures::JsFuture::from(promise).await?;

        let mapped = buffer_slice.get_mapped_range();
        let stats: &[f32] = bytemuck::cast_slice(&mapped);
        let out = stats.to_vec();
        drop(mapped);
        self.mse_staging_read_buf.unmap();

        Ok(out)
    }

    /// Read back the most recent d8 output tensor in NCHW order ([3, 256, 256]).
    ///
    /// This is intentionally separate from run_frame(): callers that only render to
    /// the canvas avoid a full 768 KiB GPU->CPU transfer.
    pub async fn read_output_nchw(&mut self) -> Result<Vec<f32>, JsValue> {
        let bytes = (3 * 256 * 256 * 4) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pix2pix_output_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.output_buf,
            0,
            &self.output_staging_read_buf,
            0,
            bytes,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.output_staging_read_buf.slice(..bytes);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| match res {
                Ok(()) => {
                    let _ = resolve.call0(&JsValue::NULL);
                }
                Err(e) => {
                    let _ = reject.call1(
                        &JsValue::NULL,
                        &JsValue::from_str(&format!("output readback map failed: {e:?}")),
                    );
                }
            });
        });
        let _ = self.device.poll(wgpu::Maintain::Poll);
        wasm_bindgen_futures::JsFuture::from(promise).await?;

        let mapped = buffer_slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&mapped);
        let out = values.to_vec();
        drop(mapped);
        self.output_staging_read_buf.unmap();
        Ok(out)
    }

    pub fn run_frame(&mut self, rgba_bytes: &[u8]) -> Result<(), JsValue> {
        if rgba_bytes.len() != 256 * 256 * 4 {
            return Err(JsValue::from_str(
                "Input image buffer must be exactly 256x256x4 bytes (262,144 bytes)",
            ));
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);

        for (idx, stage) in self.decoders.iter().enumerate() {
            if let Some(norm) = &stage.norm {
                if norm.base_params.apply_dropout != 0 {
                    let mut updated = norm.base_params;
                    updated.seed = norm
                        .base_params
                        .seed
                        .wrapping_add(self.frame_counter.wrapping_mul(0x9E37_79B9))
                        ^ (idx as u32 * 0x85EB_CA6B);
                    self.queue
                        .write_buffer(&norm.param_buf, 0, bytemuck::bytes_of(&updated));
                }
            }
        }

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );

        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Swapchain acquire error: {e:?}")))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pix2pix_frame_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unet256_compute_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.preprocess_pipeline);
            cpass.set_bind_group(0, &self.preprocess_bg, &[]);
            cpass.dispatch_workgroups(16, 16, 1);

            for (conv, norm_opt) in &self.encoders {
                cpass.set_pipeline(&self.conv_pipeline);
                cpass.set_bind_group(0, &conv.bind_group, &[]);
                let (wx, wy, wz) = conv.workgroups;
                cpass.dispatch_workgroups(wx, wy, wz);

                if let Some(norm) = norm_opt {
                    cpass.set_pipeline(&self.norm_pipeline);
                    cpass.set_bind_group(0, &norm.bind_group, &[]);
                    cpass.dispatch_workgroups(norm.workgroups_x, 1, 1);
                }
            }

            for stage in &self.decoders {
                if self.use_texture2d_array_skip {
                    if let Some(skip_stack) = &stage.skip_stack {
                        cpass.set_pipeline(&self.stack_skip_pipeline);
                        cpass.set_bind_group(0, &skip_stack.stack_bind_group, &[]);
                        let (sx, sy, sz) = skip_stack.stack_workgroups;
                        cpass.dispatch_workgroups(sx, sy, sz);

                        cpass.set_pipeline(&self.deconv_texarray_storage_pipeline);
                        cpass.set_bind_group(0, &skip_stack.deconv_texarray_bind_group, &[]);
                        let (wx, wy, wz) = stage.deconv.workgroups;
                        cpass.dispatch_workgroups(wx, wy, wz);
                    } else {
                        cpass.set_pipeline(&self.deconv_pipeline);
                        cpass.set_bind_group(0, &stage.deconv.bind_group, &[]);
                        let (wx, wy, wz) = stage.deconv.workgroups;
                        cpass.dispatch_workgroups(wx, wy, wz);
                    }
                } else {
                    cpass.set_pipeline(&self.deconv_pipeline);
                    cpass.set_bind_group(0, &stage.deconv.bind_group, &[]);
                    let (wx, wy, wz) = stage.deconv.workgroups;
                    cpass.dispatch_workgroups(wx, wy, wz);
                }

                if let Some(norm) = &stage.norm {
                    cpass.set_pipeline(&self.norm_pipeline);
                    cpass.set_bind_group(0, &norm.bind_group, &[]);
                    cpass.dispatch_workgroups(norm.workgroups_x, 1, 1);
                }
            }
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("unet256_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}
