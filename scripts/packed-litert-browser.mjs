import { createServer } from 'vite'
import { chromium } from 'playwright'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

/** Optional real-browser probe of the packed Lil Local managed LiteRT WASM runtime. */
export async function probePackedBrowser(fixtureRoot, repositoryRoot) {
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
      args: ['--no-sandbox'],
    })
    console.log('CHROMIUM', browser.version())
    const page = await browser.newPage()
    const failures = []
    page.on('pageerror', error => failures.push(error.message))
    await page.goto('http://127.0.0.1:' + port + '/src/index.ts', { waitUntil: 'domcontentloaded' })
    const result = await page.evaluate(async () => {
      const { runPackedTinyLitert } = await import('/src/packed-litert-browser.ts')
      return runPackedTinyLitert()
    })
    if (failures.length) throw Error('Browser page errors: ' + failures.join('; '))
    if (!result.cached) throw Error('Packed runtime did not reuse compiled model')
    console.log('PACKED_WASM_NUMERIC_PARITY', JSON.stringify(result))
  } finally {
    await browser?.close()
    await server.close()
  }
}
