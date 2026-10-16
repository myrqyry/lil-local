#!/usr/bin/env node
/**
 * Opt-in WebGPU matmul probe. NOT an inference execution provider.
 * LiteRT.js and ONNX Runtime Web own their internal kernels.
 */
import { createServer } from 'node:http'
import { chromium } from 'playwright'
import { selectMeasuredWorkgroup } from './webgpu-matmul-selection.mjs'

function readSize() {
  if (process.argv.length === 2) return 256
  if (process.argv.length !== 4 || process.argv[2] !== '--size') {
    throw new Error('Usage: node scripts/probe-webgpu-matmul.mjs [--size 64|128|256|512]')
  }
  const size = Number(process.argv[3])
  if (![64, 128, 256, 512].includes(size)) throw new Error('Size must be 64, 128, 256 or 512')
  return size
}

const size = readSize()
const server = createServer((_request, response) => {
  response.writeHead(200, { 'content-type': 'text/html' })
  response.end('<!doctype html><title>Lil Tart WebGPU probe</title>')
})
let browser
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_BINARY || '/usr/bin/chromium',
    args: [
      '--no-sandbox', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
      '--use-angle=vulkan', '--disable-vulkan-surface',
    ],
  })
  const page = await browser.newPage()
  await page.goto('http://127.0.0.1:' + server.address().port + '/')
  const result = await page.evaluate(async size => {
    if (!navigator.gpu) throw new Error('WebGPU unavailable on secure origin')
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('No WebGPU adapter')
    const device = await adapter.requestDevice()
    const bytes = size * size * 4
    const inputA = Float32Array.from({ length: size * size }, (_, i) => ((i % 17) - 8) / 8)
    const inputB = Float32Array.from({ length: size * size }, (_, i) => ((i % 13) - 6) / 7)
    const upload = values => {
      const buffer = device.createBuffer({
        size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(buffer, 0, values)
      return buffer
    }
    const a = upload(inputA)
    const b = upload(inputB)
    const candidates = []
    try {
      for (const groupSize of [8, 16]) {
        const module = device.createShaderModule({ code: [
          'const N: u32 = ' + size + 'u;',
          '@group(0) @binding(0) var<storage, read> a: array<f32>;',
          '@group(0) @binding(1) var<storage, read> b: array<f32>;',
          '@group(0) @binding(2) var<storage, read_write> c: array<f32>;',
          '@compute @workgroup_size(' + groupSize + ',' + groupSize + ',1)',
          'fn main(@builtin(global_invocation_id) gid: vec3<u32>) {',
          'let row=gid.y; let col=gid.x;',
          'if (row>=N || col>=N) { return; }',
          'var sum: f32=0.0;',
          'for (var k: u32=0u; k<N; k++) { sum+=a[row*N+k]*b[k*N+col]; }',
          'c[row*N+col]=sum;',
          '}',
        ].join('\n') })
        const errors = (await module.getCompilationInfo()).messages
          .filter(message => message.type === 'error')
        if (errors.length) throw new Error('WGSL compilation: ' + JSON.stringify(errors))
        const pipeline = device.createComputePipeline({
          layout: 'auto', compute: { module, entryPoint: 'main' },
        })
        const output = device.createBuffer({
          size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
        const readback = device.createBuffer({
          size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        try {
          const bindings = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: a } },
              { binding: 1, resource: { buffer: b } },
              { binding: 2, resource: { buffer: output } },
            ],
          })
          const execute = async dispatches => {
            const commands = device.createCommandEncoder()
            for (let i = 0; i < dispatches; i++) {
              const pass = commands.beginComputePass()
              pass.setPipeline(pipeline)
              pass.setBindGroup(0, bindings)
              pass.dispatchWorkgroups(Math.ceil(size / groupSize), Math.ceil(size / groupSize))
              pass.end()
            }
            commands.copyBufferToBuffer(output, 0, readback, 0, bytes)
            const start = performance.now()
            device.queue.submit([commands.finish()])
            await readback.mapAsync(GPUMapMode.READ)
            const elapsedMs = performance.now() - start
            const values = new Float32Array(readback.getMappedRange().slice(0))
            readback.unmap()
            return { elapsedMs, values }
          }
          await execute(1)
          const samplesMs = []
          let last
          for (let round = 0; round < 6; round++) {
            const sample = await execute(8)
            samplesMs.push(sample.elapsedMs / 8)
            last = sample.values
          }
          let maxAbsError = 0
          for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
              let reference = 0
              for (let k = 0; k < size; k++) {
                reference += inputA[row * size + k] * inputB[k * size + col]
              }
              maxAbsError = Math.max(
                maxAbsError, Math.abs(last[row * size + col] - reference),
              )
            }
          }
          candidates.push({
            workgroup: groupSize, samplesMs,
            maxAbsError, correct: Number.isFinite(maxAbsError) && maxAbsError < 0.003,
          })
        } finally {
          readback.destroy()
          output.destroy()
        }
      }
    } finally {
      a.destroy()
      b.destroy()
      device.destroy()
    }
    return {
      adapter: adapter.info ? {
        vendor: adapter.info.vendor, architecture: adapter.info.architecture,
        device: adapter.info.device, description: adapter.info.description,
      } : null,
      size, candidates,
      measured: 'Wall time of submit+GPU+readback divided by eight dispatches',
    }
  }, size)
  const decision = selectMeasuredWorkgroup(result.candidates)
  console.log(JSON.stringify({
    kind: 'lil-tart-webgpu-matmul-probe-v1',
    ...result, decision,
    inferenceIntegration: 'none: LiteRT and ONNX Runtime kernels unchanged',
  }, null, 2))
  if (result.candidates.some(candidate => !candidate.correct)) {
    throw new Error('Kernel output failed CPU numerical comparison')
  }
} finally {
  if (browser) await browser.close()
  if (server.listening) await new Promise(resolve => server.close(resolve))
}
