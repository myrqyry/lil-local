import { createServer } from 'vite'
import { chromium } from 'playwright'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

/** Opt-in packed-consumer numeric probe for explicit WASM or WebGPU; never falls back. */
export async function probePackedBrowser(fixtureRoot, repositoryRoot, backend = 'wasm') {
  const metadata = JSON.parse(await readFile(
    resolve(repositoryRoot, 'tests/runtime-qualification/tiny-litert-baseline/fixtures/asset.json'), 'utf8',
  ))
  const response = await fetch(metadata.url, { signal: AbortSignal.timeout(30000) })
  if (!response.ok) throw Error('Tiny model fetch HTTP ' + response.status)
  const model = Buffer.from(await response.arrayBuffer())
  const digest = createHash('sha256').update(model).digest('hex')
  if (model.length !== metadata.bytes || digest !== metadata.sha256) {
    throw Error('Pinned tiny model size/hash mismatch')
  }
  console.log('Tiny LiteRT model verified: ' + model.length + ' bytes, SHA256 ' + digest)
  const server = await createServer({
    root: fixtureRoot, configFile: resolve(fixtureRoot, 'vite.config.ts'),
    server: { host: '127.0.0.1', port: 0 },
    plugins: [{
      name: 'pinned-local-tiny-litert',
      configureServer(dev) {
        dev.middlewares.use('/_tiny/model.tflite', (_req, res) => {
          res.setHeader('Content-Type', 'application/octet-stream')
          res.setHeader('Content-Length', model.length)
          res.end(model)
        })
      },
    }],
  })
  let browser
  try {
    await server.listen()
    const port = server.httpServer.address().port
    browser = await chromium.launch({
      headless: true, executablePath: process.env.CHROMIUM_BINARY || '/usr/bin/chromium',
      args: backend === 'webgpu' ? [
        '--no-sandbox', '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
        '--use-angle=vulkan', '--disable-vulkan-surface',
      ] : ['--no-sandbox'],
    })
    console.log('CHROMIUM', browser.version())
    const page = await browser.newPage()
    const failures = []
    page.on('pageerror', error => failures.push(error.message))
    await page.goto('http://127.0.0.1:' + port + '/src/index.ts', { waitUntil: 'domcontentloaded' })
    const result = await page.evaluate(async backend => {
      if (backend === 'webgpu') {
        const adapter = await navigator.gpu?.requestAdapter()
        if (!adapter) throw new Error('WebGPU adapter unavailable')
        const info = adapter.info
        console.log('WEBGPU_ADAPTER', JSON.stringify({ vendor: info?.vendor, architecture: info?.architecture, device: info?.device }))
        if (!info?.vendor || info.vendor.toLowerCase() === 'google' || info.architecture?.toLowerCase().includes('swiftshader')) {
          throw new Error('WebGPU hardware adapter not verified (software/unknown adapter)')
        }
        const device = await adapter.requestDevice()
        if (!device) throw new Error('WebGPU device unavailable')
      }
      const { runPackedTinyLitert } = await import('/src/packed-litert-browser.ts')
      return runPackedTinyLitert(backend)
    }, backend)
    if (failures.length) throw Error('Browser page errors: ' + failures.join('; '))
    if (!result.cached) throw Error('Packed runtime did not reuse compiled model')
    console.log('PACKED_' + backend.toUpperCase() + '_NUMERIC_PARITY', JSON.stringify(result))
  } finally {
    await browser?.close()
    await server.close()
  }
}
