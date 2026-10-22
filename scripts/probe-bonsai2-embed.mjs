#!/usr/bin/env node
/**
 * Real local-only opt-in ONNX WebGPU embedding graph qualifier.
 * Model blobs stay outside the repository, served from an isolated local
 * HTTP route; the real browser runtime runs on the NVIDIA GPU if available.
 */
import { createServer } from 'vite'
import { chromium } from 'playwright'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { resolve } from 'node:path'

// No implicit GPU probe or model loading. Even the isolated embed test is
// expensive and does NOT establish decoder usability on the pinned Scan graph.
const inspectOnly = process.argv.includes('--inspect-adapter')
const allowEmbed = process.argv.includes('--run-embedding')
if (!inspectOnly && !allowEmbed) {
  console.log('Bonsai 2B decoder has 18 Scan nodes: this ONNX export is blocked for browser text generation.')
  console.log('Opt in: --inspect-adapter (feature check only) or --run-embedding (isolated graph test, explicit large model assets required).')
  process.exit(0)
}

const root = resolve('apps/playground')
const assets = '/tmp/lil-local-ternary-bonsai-inspect'
const paths = new Map([
  ['embed_tokens_q2f16.onnx', 'embed.onnx'],
  ['embed_tokens_q2f16.onnx_data', 'embed.onnx_data'],
])
const server = await createServer({
  root,
  server: { host: '127.0.0.1', port: 0 },
  plugins: [{
    name: 'private-bonsai-model-probe-files',
    configureServer(dev) {
      dev.middlewares.use('/_bonsai/', (req, res, next) => {
        const name = decodeURIComponent((req.url || '').split('?')[0]).replace(/^\/+/, '')
        const file = paths.get(name)
        if (!file) return next()
        const target = resolve(assets, file)
        if (!existsSync(target)) {
          res.statusCode = 404
          return res.end('Model asset not downloaded')
        }
        const size = statSync(target).size
        res.setHeader('Content-Type', 'application/octet-stream')
        res.setHeader('Content-Length', size)
        res.setHeader('Cache-Control', 'no-store')
        res.statusCode = 200
        createReadStream(target).pipe(res)
      })
    },
  }],
})
let browser
try {
  await server.listen()
  const port = server.httpServer.address().port
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
  const assetRequests = []
  page.on('request', request => {
    if (request.url().includes('/_bonsai/')) assetRequests.push(request.url())
  })
  const warnings = []
  page.on('console', msg => { if (msg.type() === 'warning' || msg.type() === 'error') warnings.push(msg.text().slice(0,500)) })
  page.on('pageerror', error => warnings.push('PAGE ERROR ' + error.message))
  await page.goto('http://127.0.0.1:' + port + '/', {waitUntil: 'domcontentloaded', timeout: 20000})
  const eligibility = await page.evaluate(async () => {
    const module = await import('/src/substrates/bonsai2Onnx.ts')
    return await module.inspectBonsaiGpu()
  })
  console.log('WEBGPU PREREQUISITE', JSON.stringify(eligibility))
  if (!eligibility.shaderF16) {
    if (assetRequests.length) throw new Error('Bonsai model assets downloaded despite failed preflight')
    console.log('BLOCKED_BEFORE_WEIGHTS: shader-f16 unavailable; no ORT session or model asset request attempted')
  } else if (inspectOnly) {
    console.log('INSPECTION_ONLY: no model graph loaded or downloaded.')
  } else {
    const result = await Promise.race([
      page.evaluate(async () => {
        const module = await import('/src/substrates/bonsai2Onnx.ts')
        return await module.probeBonsaiEmbedding(
          '/_bonsai/embed_tokens_q2f16.onnx',
          '/_bonsai/embed_tokens_q2f16.onnx_data',
        )
      }),
      new Promise((_resolve, reject) => setTimeout(() => reject(Error('Embedding inference timed out at 150s')),150000)),
    ])
    console.log('ACTUAL EMBED WEBGPU PROBE',JSON.stringify(result))
  }
  console.log('MODEL ASSET REQUESTS', JSON.stringify(assetRequests))
  console.log('WARNINGS',JSON.stringify(warnings.slice(0,22)))
} finally {
  if (browser) await browser.close()
  await server.close()
}
