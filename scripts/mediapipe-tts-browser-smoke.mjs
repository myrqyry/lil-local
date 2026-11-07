// Execute the *actual Lil Local public TTS engine* in Chromium with a caller-selected local Phonon bundle.
// Set LIL_LOCAL_TTS_MODEL to an existing official phonon_int8.litertlm path.
// No automatic model downloads and no microphone access.
import { createServer } from 'vite'
import { chromium } from 'playwright'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { resolve, join, basename } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const modelPath = process.env.LIL_LOCAL_TTS_MODEL
if (!modelPath || !existsSync(modelPath) || basename(modelPath) !== 'phonon_int8.litertlm') {
  throw new Error('Set LIL_LOCAL_TTS_MODEL to an existing phonon_int8.litertlm (no implicit download)')
}
const wasmRoot = resolve(root, 'node_modules/.pnpm/@mediapipe+tasks-audio@1.2.2/node_modules/@mediapipe/tasks-audio/wasm')
const port = 4396
const server = await createServer({
  root,
  configFile: false,
  server: { host: '127.0.0.1', port, strictPort: true },
  plugins: [{
    name: 'qualification-assets',
    configureServer(vite) {
      vite.middlewares.use((request, response, next) => {
        const url = decodeURIComponent(request.url?.split('?')[0] ?? '')
        if (!url.startsWith('/__tts_assets/')) return next()
        const requested = url.slice('/__tts_assets/'.length)
        let path
        if (requested === 'phonon_int8.litertlm') path = modelPath
        else if (/^wasm\/audio_wasm_(?:internal|module_internal)\.(?:js|wasm)$/.test(requested)) {
          path = join(wasmRoot, basename(requested))
        } else {
          response.writeHead(404).end()
          return
        }
        if (!existsSync(path)) { response.writeHead(404).end(); return }
        response.writeHead(200, {
          'Content-Type': path.endsWith('.wasm') ? 'application/wasm'
            : path.endsWith('.litertlm') ? 'application/octet-stream' : 'text/javascript',
          'Content-Length': String(statSync(path).size),
          'Cache-Control': 'no-store',
        })
        createReadStream(path).pipe(response)
      })
    },
  }],
})
let browser
try {
  await server.listen()
  browser = await chromium.launch({executablePath:'/usr/bin/chromium', headless:true, args:['--no-sandbox','--disable-dev-shm-usage']})
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('http://127.0.0.1:' + port + '/tests/fixtures/mediapipe-tts-qualification.html')
  const output = await page.evaluate(async () => {
    return await window.runLilLocalTtsQualification()
  })
  console.log(JSON.stringify(output, null, 2))
  if (errors.length) throw new Error('Browser exceptions: ' + errors.join('; '))
  if (output.sampleRateHz !== 24000 || output.samples < 2400 || output.peak < 0.001 || output.rms < 0.0001) {
    throw new Error('Output missing or silent / invalid: ' + JSON.stringify(output))
  }
  if (!output.voiceIds.some(voice => voice.includes('freya'))) throw new Error('Phonon freya voice missing')
  if (output.streamedChunks < 2 || !output.sawFinalChunk || output.streamedSamples < 2400) throw new Error('Streaming synthesis incomplete: ' + JSON.stringify(output))
  if (output.switchedSamples < 2400 || output.switchedPeak < 0.001) throw new Error('Voice switch produced no speech')
  console.log('PASS: Lil Local MediaPipe TTS public engine -> real Phonon Chromium WASM -> non-silent PCM')
} finally {
  await browser?.close()
  await server.close()
}
