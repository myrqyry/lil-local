#!/usr/bin/env node
/** Real Chromium smoke of the packed-public-style Lil Local SlimSAM API.
 * Usage: pnpm test:slimsam:browser /path/to/image.png 100 80
 * The image remains local. Model inference uses revision-pinned HF weights.
 */
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { extname, resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const root = fileURLToPath(new URL('../', import.meta.url))
const [imagePath, rawX, rawY] = process.argv.slice(2)
const x = Number(rawX), y = Number(rawY)
const mime = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' }[extname(imagePath ?? '').toLowerCase()]
if (!imagePath || !mime || !Number.isFinite(x) || !Number.isFinite(y)) {
  console.error('Usage: pnpm test:slimsam:browser /absolute/image.png pointX pointY')
  process.exit(2)
}
const bytes = Array.from(await readFile(resolve(imagePath)))
const port = await new Promise((ok, fail) => {
  const socket = createServer().on('error', fail)
  socket.listen(0, '127.0.0.1', () => {
    const chosen = socket.address().port
    socket.close(() => ok(chosen))
  })
})
const base = `http://127.0.0.1:${port}`
const server = spawn('pnpm', ['--filter', 'playground', 'exec', 'vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
  cwd: root, detached: true, stdio: 'ignore',
})
let browser
try {
  let ready = false
  for (let attempt = 0; attempt < 120; attempt++) {
    if (server.exitCode !== null) throw new Error('Vite exited before becoming available')
    try { ready = (await fetch(base, { signal: AbortSignal.timeout(500) })).ok } catch { /* startup */ }
    if (ready) break
    await delay(250)
  }
  if (!ready) throw new Error('Vite did not start')
  const binary = [process.env.CHROME_PATH, '/usr/bin/chromium', '/usr/bin/google-chrome-beta', chromium.executablePath()]
    .find(candidate => candidate && existsSync(candidate))
  if (!binary) throw new Error('Chromium not installed; set CHROME_PATH to a Chromium executable')
  browser = await chromium.launch({
    headless: true, executablePath: binary,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'],
  })
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', err => errors.push(err.message))
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  await page.goto(base, { waitUntil: 'domcontentloaded' })
  const runtimeModule = '/@fs' + resolve(root, 'packages/image-segmentation/src/runtime.ts')
  const result = await page.evaluate(async ({ bytes, x, y, mime, runtimeModule }) => {
    const { selectSlimSamPoint } = await import(runtimeModule)
    const image = new Blob([Uint8Array.from(bytes)], { type: mime })
    const start = performance.now()
    const regions = await selectSlimSamPoint(image, { x, y })
    return {
      elapsedMs: Math.round(performance.now() - start),
      regions: regions.map(r => ({
        width: r.width, height: r.height,
        predictedQuality: r.score,
        selectedPixels: r.data.filter(value => value > 0).length,
      })),
    }
  }, { bytes, x, y, mime, runtimeModule })
  if (!result.regions.length || !result.regions.some(r => r.selectedPixels > 0)) {
    throw new Error('Real model returned no nonempty object masks')
  }
  if (errors.length) throw new Error('Browser errors: ' + JSON.stringify(errors))
  console.log(JSON.stringify({ model: 'Xenova/slimsam-77-uniform', quantization: 'q8', runtime: 'Lil Local / Transformers.js 4 / WASM', ...result, browserErrors: 0 }, null, 2))
} finally {
  await browser?.close()
  try { process.kill(-server.pid, 'SIGTERM') } catch { /* exited already */ }
}
