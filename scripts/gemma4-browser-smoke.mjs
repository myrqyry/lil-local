#!/usr/bin/env node
/** Real Gemma 4 E2B/E4B browser proof, via Lil Local's production UI and worker.
 * Explicit local SHA-pinned model required; never downloads weights implicitly.
 * Set LIL_LOCAL_GEMMA_WEB_MODEL or LIL_LOCAL_GEMMA_PROBE_ONLY=1.
 */
import {spawn} from 'node:child_process'
import {createHash} from 'node:crypto'
import {createReadStream, existsSync, statSync} from 'node:fs'
import {createServer} from 'node:net'
import {resolve, basename} from 'node:path'
import {fileURLToPath} from 'node:url'
import {setTimeout as delay} from 'node:timers/promises'
import {chromium} from 'playwright'

const root = fileURLToPath(new URL('../', import.meta.url))
const model = process.env.LIL_LOCAL_GEMMA_WEB_MODEL
const probeOnly = process.env.LIL_LOCAL_GEMMA_PROBE_ONLY === '1'
const variant = (process.env.LIL_LOCAL_GEMMA_VARIANT || 'e2b').toLowerCase()
const checks = {
  e2b: {
    name: 'gemma-4-E2B-it-web.litertlm',
    bytes: 2_008_432_640,
    sha256: '3a08e8d94e23b814ae5414469c370c503813949acb8ceaa17e4ebf8a35af35b5',
  },
  e4b: {
    name: 'gemma-4-E4B-it-web.litertlm',
    bytes: 2_969_059_328,
    sha256: '3904d826d5dddd25ea173e85204caec09e68ba038116e9b992b69cbdc94f57a0',
  },
}
if (!(variant in checks)) throw new Error('LIL_LOCAL_GEMMA_VARIANT must be e2b or e4b')
const expectedName = checks[variant].name
const expectedBytes = checks[variant].bytes
const expectedDigest = checks[variant].sha256
if (!probeOnly && (!model || !existsSync(model) || basename(model) !== expectedName)) {
  throw new Error('Set LIL_LOCAL_GEMMA_WEB_MODEL to a downloaded, pinned ' + expectedName)
}
if (!probeOnly) {
  if (statSync(model).size !== expectedBytes) throw new Error('Wrong Gemma web checkpoint size')
  const digest = await new Promise((resolveHash, reject) => {
    const hash = createHash('sha256')
    const stream = createReadStream(model)
    stream.on('data', chunk => hash.update(chunk))
    stream.on('error', reject)
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
  if (digest !== expectedDigest) throw new Error('Gemma checkpoint SHA-256 does not match the pinned source')
  console.log('Verified official '+variant.toUpperCase()+' WebGPU model: ' + expectedBytes + ' bytes; SHA-256 ' + digest)
}
const port=await new Promise((ok, fail) => {
  const socket=createServer().on('error', fail)
  socket.listen(0,'127.0.0.1',() => {
    const selected=socket.address().port
    socket.close(()=>ok(selected))
  })
})
const origin='http://127.0.0.1:'+port
const server=spawn('pnpm',['--filter','playground','exec','vite','preview','--host','127.0.0.1','--port',String(port),'--strictPort'],{
  cwd:root,detached:true,stdio:'ignore',
})
let browser
const pageErrors=[]
const requests=[]
const browserArgs=['--no-sandbox','--disable-dev-shm-usage','--ignore-gpu-blocklist','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU']
try {
  let ready=false
  for(let i=0;i<120;i++){
    if(server.exitCode!==null)throw new Error('Production preview exited')
    try{ready=(await fetch(origin,{signal:AbortSignal.timeout(500)})).ok}catch{}
    if(ready)break
    await delay(250)
  }
  if(!ready)throw new Error('Production preview unavailable')
  browser=await chromium.launch({
    headless:true,executablePath:process.env.CHROME_PATH||'/usr/bin/chromium',args:browserArgs,
  })
  const page=await browser.newPage({viewport:{width:1280,height:900}})
  page.on('pageerror',err=>pageErrors.push(err.message))
  page.context().on('request',request=>requests.push(request.url()))
  page.on('console',entry=>{
    if(entry.type()==='error')console.log('BROWSER_CONSOLE_ERROR '+entry.text().slice(0,500))
  })
  await page.goto(origin,{waitUntil:'domcontentloaded'})
  await page.locator('button[aria-label="Open Gemma 4 · Browser Inference"]').waitFor({timeout:20000})
  if(requests.some(url=>url.includes('gemma-4-E2B-it')||url.includes('gemma-4-E4B-it')))throw new Error('Checkpoint fetched before user action')
  await page.locator('button[aria-label="Open Gemma 4 · Browser Inference"]').click()
  await page.locator('[aria-label="Gemma 4 browser inference"]').waitFor({timeout:20000})
  const gpu=await page.evaluate(async()=>{
    const adapter=await navigator.gpu?.requestAdapter()
    return adapter?{found:true,info:adapter.info||null}: {found:false}
  })
  console.log('WEBGPU_ADAPTER',JSON.stringify(gpu))
  if(!gpu.found)throw new Error('No browser WebGPU adapter was exposed; cannot qualify GPU inference on this browser')
  const wasm='/vendor/litert-lm-0.19.0/litertlm_wasm_internal.wasm'
  const wasmResponse=await fetch(origin+wasm,{method:'HEAD'})
  if(!wasmResponse.ok)throw new Error('Self-hosted LiteRT-LM WASM missing: '+wasmResponse.status)
  console.log('LOCAL_WASM',wasmResponse.status,wasmResponse.headers.get('content-length'))
  if(probeOnly){
    console.log('PASS: Gemma card renders, no checkpoint auto-fetch, WebGPU available, WASM self-hosted')
  }else{
    await page.locator('[aria-label="Gemma model"]').selectOption(variant)
    const cdp=await page.context().newCDPSession(page)
    const doc=await cdp.send('DOM.getDocument')
    const selector='input[aria-label="Gemma local model file"]'
    const {nodeId}=await cdp.send('DOM.querySelector',{nodeId:doc.root.nodeId,selector})
    if(!nodeId)throw new Error('Local Gemma model file chooser unavailable')
    await cdp.send('DOM.setFileInputFiles',{nodeId,files:[resolve(model)]})
    await page.getByText('Selected local file: '+expectedName,{exact:false}).waitFor({timeout:15000})
    const began=Date.now()
    await page.locator('button[aria-label="Load Gemma locally"]').click()
    await page.waitForFunction(
      ()=>{
        const text=document.querySelector('[role="status"]')?.textContent||''
        return text.includes('Gemma status: ready')||text.includes('Gemma status: error')
      },null,{timeout:480000},
    )
    const loadStatus=await page.locator('[role="status"]').textContent()
    if(!loadStatus.includes('ready'))throw new Error('Real Gemma model load failed: '+loadStatus+' / '+(await page.locator('[role="alert"]').allTextContents()).join(' | '))
    console.log('GEMMA_LOADED',Date.now()-began,'ms',variant)
    await page.locator('textarea[aria-label="Gemma prompt"]').fill('In one short sentence, what is the capital of France?')
    await page.locator('input[aria-label="Gemma max output tokens"]').fill('32')
    await page.locator('button[aria-label="Generate with Gemma locally"]').click()
    await page.waitForFunction(
      ()=>{
        const text=document.querySelector('[role="status"]')?.textContent||''
        return text.includes('Gemma status: ready')||text.includes('Gemma status: error')
      },null,{timeout:240000},
    )
    const output=await page.locator('[aria-label="Gemma model output"]').textContent()
    const alerts=await page.locator('[role="alert"]').allTextContents()
    if(alerts.length || !output?.trim() || !output.toLowerCase().includes('paris')){
      throw new Error('Real Gemma '+variant+' text generation incomplete: '+JSON.stringify({output,alerts}))
    }
    console.log('REAL_GEMMA_OUTPUT',JSON.stringify(output))
    await page.getByRole('button',{name:'Unload / stop'}).click()
    if(!(await page.locator('[role="status"]').textContent()).includes('idle'))throw new Error('Gemma unload failed')
    console.log('PASS: SHA-pinned Gemma 4 '+variant+' WebGPU -> actual worker load -> native text generation -> unload')
  }
  if(pageErrors.length)throw new Error('Page exceptions: '+pageErrors.join(' | '))
  if(requests.some(url=>url.includes('cdn.jsdelivr.net/npm/@litert-lm/core/wasm'))){
    throw new Error('WASM loaded from CDN instead of locally hosted runtime')
  }
}finally{
  await browser?.close()
  try{process.kill(-server.pid,'SIGTERM')}catch{}
}
