#!/usr/bin/env node
/** Real Gemma 4 E2B/E4B browser proof, via Lil Local's production UI and worker.
 * Explicit local SHA-pinned model required; never downloads weights implicitly.
 * Set LIL_LOCAL_GEMMA_WEB_MODEL or LIL_LOCAL_GEMMA_PROBE_ONLY=1.
 */
import {spawn, spawnSync} from 'node:child_process'
import {createHash} from 'node:crypto'
import {createReadStream, existsSync, statSync, mkdtempSync, readFileSync, rmSync} from 'node:fs'
import {createServer} from 'node:net'
import {resolve, basename, join} from 'node:path'
import {tmpdir} from 'node:os'
import {fileURLToPath} from 'node:url'
import {setTimeout as delay} from 'node:timers/promises'
import {chromium} from 'playwright'

const root = fileURLToPath(new URL('../', import.meta.url))
const model = process.env.LIL_LOCAL_GEMMA_WEB_MODEL
const probeOnly = process.env.LIL_LOCAL_GEMMA_PROBE_ONLY === '1'
const variant = (process.env.LIL_LOCAL_GEMMA_VARIANT || 'e2b').toLowerCase()
const checks = {
  full: {
    name: 'gemma-4-E2B-it.litertlm',
    bytes: 2_588_147_712,
    sha256: '181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c',
  },
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
if (!(variant in checks)) throw new Error('LIL_LOCAL_GEMMA_VARIANT must be full, e2b or e4b')
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
    await page.locator('textarea[aria-label="Gemma prompt"]').fill(
      'Remember this exact three-word code: turquoise railway mushroom. What is the capital of France? Answer only the capital.'
    )
    await page.locator('input[aria-label="Gemma max output tokens"]').fill('96')
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
    const initialStats=await page.locator('[aria-label="Gemma inference stats"]').allTextContents()
    console.log('NATIVE_CHAT_STATS_FIRST',JSON.stringify(initialStats))
    // The second prompt deliberately lacks the code, so it cannot pass
    // unless the native conversation retains the first user's message.
    await page.locator('textarea[aria-label="Gemma prompt"]').fill(
      'What exact three-word code did I ask you to remember in my previous message? Answer only the code.'
    )
    await page.locator('button[aria-label="Generate with Gemma locally"]').click()
    await page.waitForFunction(()=>{
      const status=document.querySelector('[role="status"]')?.textContent||''
      const history=document.querySelector('[aria-label="Gemma chat history"]')?.textContent||''
      return status.includes('Gemma status: ready') &&
        history.includes('What exact three-word code') &&
        (document.querySelector('[aria-label="Gemma model output"]')?.textContent||'').length>0
    },null,{timeout:240000})
    const remembered=await page.locator('[aria-label="Gemma model output"]').textContent()
    const newAlerts=await page.locator('[role="alert"]').allTextContents()
    if(newAlerts.length || !remembered?.toLowerCase().includes('turquoise railway mushroom')){
      throw new Error('Native chat did not remember the previous prompt: '+JSON.stringify({remembered,newAlerts}))
    }
    console.log('REAL_GEMMA_FOLLOWUP',JSON.stringify(remembered))
    const followupStats=await page.locator('[aria-label="Gemma inference stats"]').allTextContents()
    console.log('NATIVE_CHAT_STATS_FOLLOWUP',JSON.stringify(followupStats))
    await page.locator('button[aria-label="New Gemma chat"]').click()
    if((await page.locator('[aria-label="Gemma chat history"]').textContent())?.trim()){
      throw new Error('New chat left prior messages visible')
    }
    console.log('NEW_CHAT_CLEARED')
    if(variant === 'full'){
      // Exercise the same full checkpoint and engine settings that previously
      // worked through PodQast, but now via Lil Local's own public UI.
      const png = await page.evaluate(()=>{
        const canvas = document.createElement('canvas')
        canvas.width = canvas.height = 64
        const ctx = canvas.getContext('2d')
        ctx.fillStyle = '#ffff00'
        ctx.fillRect(0,0,64,64)
        return canvas.toDataURL('image/png').split(',')[1]
      })
      await page.locator('[aria-label="Gemma image input"]').setInputFiles({
        name:'yellow-square.png',mimeType:'image/png',buffer:Buffer.from(png,'base64'),
      })
      await page.locator('textarea[aria-label="Gemma prompt"]').fill(
        'What color is the filled square in the image? Answer one color word.'
      )
      await page.locator('button[aria-label="Generate with Gemma locally"]').click()
      await page.waitForFunction(()=>{
        const status = document.querySelector('[role="status"]')?.textContent || ''
        return status.includes('Gemma status: ready') &&
          Boolean((document.querySelector('[aria-label="Gemma model output"]')?.textContent || '').trim())
      },null,{timeout:240000})
      const imageAnswer = await page.locator('[aria-label="Gemma model output"]').textContent()
      const imageErrors = await page.locator('[role="alert"]').allTextContents()
      if(imageErrors.length || !imageAnswer?.toLowerCase().includes('yellow')){
        throw new Error('Full E2B real image input failed: '+JSON.stringify({imageAnswer,imageErrors}))
      }
      console.log('REAL_GEMMA_IMAGE',JSON.stringify(imageAnswer))
      await page.locator('button[aria-label="New Gemma chat"]').click()
      // Produce actual uncompressed 16k mono PCM16 WAV, exactly as the
      // previously qualified PodQast fixture (no STT or intermediary model).
      const voice = spawnSync('espeak-ng',['-v','en-us','-s','133','--stdout','The secret word is banana.'],{
        maxBuffer:8_000_000,
      })
      if(voice.status!==0)throw new Error('espeak-ng fixture failed: '+voice.stderr?.toString())
      const wavDir = mkdtempSync(join(tmpdir(),'lil-local-gemma-audio-'))
      try {
        const wavPath=join(wavDir,'banana.wav')
        // Real-file FFmpeg output has finalized RIFF/data lengths; pipe:1
        // writes 0xffffffff placeholders and is not a canonical WAV file.
        const wav=spawnSync('ffmpeg',[
          '-nostdin','-hide_banner','-loglevel','error','-y','-i','pipe:0',
          '-ar','16000','-ac','1','-c:a','pcm_s16le',wavPath,
        ],{input:voice.stdout,maxBuffer:8_000_000})
        if(wav.status!==0)throw new Error('ffmpeg fixture failed: '+wav.stderr?.toString())
        await page.locator('[aria-label="Gemma WAV audio input"]').setInputFiles({
          name:'banana.wav',mimeType:'audio/wav',buffer:readFileSync(wavPath),
        })
      } finally {
        rmSync(wavDir,{recursive:true,force:true})
      }
      await page.locator('textarea[aria-label="Gemma prompt"]').fill(
        'Listen to the attached audio. Transcribe exactly what words were spoken.'
      )
      await page.locator('button[aria-label="Generate with Gemma locally"]').click()
      await page.waitForFunction(()=>{
        const status = document.querySelector('[role="status"]')?.textContent || ''
        const alert = document.querySelector('[role="alert"]')?.textContent || ''
        return Boolean(alert.trim()) || (status.includes('Gemma status: ready') &&
          Boolean((document.querySelector('[aria-label="Gemma model output"]')?.textContent || '').trim()))
      },null,{timeout:240000})
      const audioAnswer = await page.locator('[aria-label="Gemma model output"]').textContent()
      const audioErrors = await page.locator('[role="alert"]').allTextContents()
      if(audioErrors.length || !audioAnswer?.toLowerCase().includes('banana')){
        throw new Error('Full E2B real audio input failed: '+JSON.stringify({audioAnswer,audioErrors}))
      }
      console.log('REAL_GEMMA_AUDIO',JSON.stringify(audioAnswer))
      await page.locator('button[aria-label="New Gemma chat"]').click()
    }
    if(process.env.LIL_LOCAL_GEMMA_THINKING_PROBE==='1'){
      await page.locator('[aria-label="Enable thinking"]').check()
      await page.locator('input[aria-label="Gemma max output tokens"]').fill('256')
      await page.locator('textarea[aria-label="Gemma prompt"]').fill('What is two plus two? Reply with the number only.')
      await page.locator('button[aria-label="Generate with Gemma locally"]').click()
      await page.waitForFunction(()=>{
        const status=document.querySelector('[role="status"]')?.textContent||''
        const history=document.querySelector('[aria-label="Gemma chat history"]')?.textContent||''
        return status.includes('Gemma status: ready') && history.includes('What is two plus two?') &&
          (document.querySelector('[aria-label="Gemma model output"]')?.textContent||'').length>0
      },null,{timeout:240000})
      const thinkingOutput=await page.locator('[aria-label="Gemma model output"]').textContent()
      const thinkingErrors=await page.locator('[role="alert"]').allTextContents()
      const reasoning=await page.getByText('Reasoning channel',{exact:true}).allTextContents()
      if(thinkingErrors.length || !thinkingOutput?.includes('4')){
        throw new Error('Native thinking-mode generation failed: '+JSON.stringify({thinkingOutput,thinkingErrors}))
      }
      console.log('THINKING_MODE_OUTPUT',JSON.stringify({thinkingOutput,reasoningVisible:reasoning.length>0}))
    }
    await page.getByRole('button',{name:'Unload / stop'}).click()
    if(!(await page.locator('[role="status"]').textContent()).includes('idle'))throw new Error('Gemma unload failed')
    console.log('PASS: SHA-pinned Gemma 4 '+variant+' WebGPU -> native multi-turn memory -> new chat -> unload')
  }
  if(pageErrors.length)throw new Error('Page exceptions: '+pageErrors.join(' | '))
  if(requests.some(url=>url.includes('cdn.jsdelivr.net/npm/@litert-lm/core/wasm'))){
    throw new Error('WASM loaded from CDN instead of locally hosted runtime')
  }
}finally{
  await browser?.close()
  try{process.kill(-server.pid,'SIGTERM')}catch{}
}
