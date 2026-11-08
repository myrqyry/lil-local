#!/usr/bin/env node
/** Real browser qualification of the two new Live Vision models, through the existing UI. */
import {spawn} from 'node:child_process'
import {createHash} from 'node:crypto'
import {createServer} from 'node:net'
import {existsSync} from 'node:fs'
import {resolve} from 'node:path'
import {fileURLToPath} from 'node:url'
import {setTimeout as delay} from 'node:timers/promises'
import {chromium} from 'playwright'

const root=fileURLToPath(new URL('../',import.meta.url))
const port=await new Promise((ok,fail)=>{
 const sock=createServer().on('error',fail)
 sock.listen(0,'127.0.0.1',()=>{
  const current=sock.address().port
  sock.close(()=>ok(current))
 })
})
const origin=`http://127.0.0.1:${port}`
const server=spawn('pnpm',['--filter','playground','exec','vite','preview','--host','127.0.0.1','--port',String(port),'--strictPort'],{
 cwd:root,detached:true,stdio:'ignore',
})
let browser
const responses=[],requestErrors=[],pageErrors=[]
try {
 let served=false
 for(let i=0;i<120;i++){
  if(server.exitCode!==null)throw new Error('Lil Local preview exited')
  try{served=(await fetch(origin,{signal:AbortSignal.timeout(500)})).ok}catch{}
  if(served)break
  await delay(250)
 }
 if(!served)throw new Error('Lil Local preview unavailable')
 const binary=[process.env.CHROME_PATH,'/usr/bin/chromium','/usr/bin/google-chrome-beta',chromium.executablePath()]
   .find(candidate=>candidate && existsSync(candidate))
 if(!binary)throw new Error('No Chromium browser executable found')
 browser=await chromium.launch({headless:true,executablePath:binary,args:['--no-sandbox','--disable-dev-shm-usage','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--enable-unsafe-swiftshader']})
 const page=await browser.newPage()
 page.on('pageerror',e=>pageErrors.push(e.message))
 page.on('requestfailed',req=>requestErrors.push(`${req.failure()?.errorText} ${req.url()}`))
 page.on('response',response=>responses.push({url:response.url(),status:response.status()}))
 await page.goto(origin,{waitUntil:'domcontentloaded'})
 const card=page.locator('button[aria-label="Open Live Vision · Ten Local Processors"]')
 await card.waitFor({timeout:20000})
 await card.click()
 await page.locator('#vision-task').waitFor()

 // SHA-pinned publicly available human reference image, never a user's photo.
 const reference='https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples-web/6887f6399745dc549dadd0022c31fc9ab01f26d6/public/pose_model.png'
 const resource=await fetch(reference,{signal:AbortSignal.timeout(30000)})
 if(!resource.ok)throw new Error('Reference photo HTTP '+resource.status)
 const bytes=Buffer.from(await resource.arrayBuffer())
 const sha=createHash('sha256').update(bytes).digest('hex')
 if(sha!=='884e822bf3f1585fdfdf1f36c06c47ceec3d745757cc14abac9cacf45353de63')throw new Error('Reference photo digest mismatch')
 const photo={name:'pose_model.png',mimeType:'image/png',buffer:bytes}
 const imageField=page.locator('input[aria-label="Image for local vision analysis"]')

 async function infer(kind,canvasLabel){
  await page.locator('#vision-task').selectOption(kind)
  await imageField.setInputFiles(photo)
  await page.getByRole('button',{name:'Analyze selected image locally'}).click()
  await page.waitForFunction(()=>[ 'Image analyzed locally','Image analysis failed'].includes(document.querySelector('[role=status]')?.textContent||''),null,{timeout:120000})
  const status=await page.locator('[role=status]').textContent()
  if(status!=='Image analyzed locally')throw new Error(kind+' failed: '+await page.locator('[role=alert]').allTextContents())
  const result=await page.evaluate(label=>{
   const canvas=document.querySelector(`canvas[aria-label="${label}"]`)
   if(!canvas)return null
   const data=canvas.getContext('2d')?.getImageData(0,0,canvas.width,canvas.height).data
   if(!data)return null
   let min=255,max=0,opaque=0
   for(let i=0;i<data.length;i+=4){min=Math.min(min,data[i]);max=Math.max(max,data[i]);if(data[i+3]===255)opaque++}
   return {width:canvas.width,height:canvas.height,min,max,opaque,total:canvas.width*canvas.height}
  },canvasLabel)
  if(!result||result.total===0||result.opaque!==result.total||result.max<=result.min)
   throw new Error(kind+' produced missing/constant/transparent output: '+JSON.stringify(result))
  return result
 }

 const person=await infer('person','Person foreground confidence mask')
 if(!responses.some(r=>r.url.includes('selfie_segmenter.tflite')&&r.status===200))
  throw new Error('Selfie segmenter weights were not loaded')
 console.log('PERSON',person)

 const depth=await infer('depthV2','Depth grayscale conditioning map')
 if(!responses.some(r=>r.url.includes('onnx-community/depth-anything-v2-small')&&r.url.includes('4472b7362082ad9968fee890ca0f1e5aca36b93d')))
  throw new Error('Pinned depth model was not requested')
 const runtimeFiles=['/ort/ort-wasm-simd-threaded.asyncify.mjs','/ort/ort-wasm-simd-threaded.asyncify.wasm']
 for(const file of runtimeFiles)if(!responses.some(r=>r.url===origin+file&&r.status===200))
   throw new Error('Missing self-hosted ONNX runtime '+file)
 if(responses.some(r=>r.url.includes('cdn.jsdelivr.net/npm/onnxruntime-web')))
  throw new Error('Depth path fetched runtime script from CDN')
 console.log('DEPTH_Q8',depth,await page.locator('[aria-label="Vision detections"]').textContent())

 // Actual still-image runs must reach the bounded session ledger and split view.
 const ledger=page.locator('[aria-label="Experiment run ledger"]')
 await page.waitForFunction(() => {
   const content=document.querySelector('[aria-label="Experiment run ledger"]')?.textContent || ''
   return content.includes('#1') && content.includes('#2')
 },null,{timeout:20000})
 const ledgerText=await ledger.textContent()
 if(!ledgerText.includes('person')||!ledgerText.includes('depthV2'))
   throw new Error('Missing real model evidence from experiment ledger: '+ledgerText)
 await page.locator('[aria-label="A/B comparison split"]').waitFor({timeout:10000})
 if(!await page.locator('canvas[aria-label="Output of run 1"]').count())
   throw new Error('Stored model output not painted in comparison stage')
 const brightness=async()=>await page.evaluate(()=>{
   const canvas=document.querySelector('canvas[aria-label="Output of run 1"]')
   if(!(canvas instanceof HTMLCanvasElement))throw new Error('Run A output canvas unavailable')
   const data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data
   let white=0
   for(let i=0;i<data.length;i+=4)if(data[i]>128)white++
   return {white,total:canvas.width*canvas.height,width:canvas.width,height:canvas.height}
 })
 await page.locator('[aria-label="Foreground display mode"]').selectOption('cutoff')
 const cutoff=page.locator('[aria-label="Foreground confidence threshold"]')
 await cutoff.focus()
 await cutoff.press('Home')
 await page.waitForFunction(()=>document.body.textContent.includes('Confidence ≥ 0.00'),null,{timeout:10000})
 const allWhite=await brightness()
 await cutoff.press('End')
 await page.waitForFunction(()=>document.body.textContent.includes('Confidence ≥ 1.00'),null,{timeout:10000})
 const strict=await brightness()
 if(allWhite.white!==allWhite.total || strict.white>=allWhite.white)
   throw new Error('Stored float confidence threshold failed to alter mask without model inference: '+JSON.stringify({allWhite,strict}))
 console.log('EXPERIMENT', {ledger:ledgerText.slice(0,180),allWhite,strict})

 // Same existing camera lifecycle: real MediaPipe person-mask VIDEO inference, mocked camera device.
 await page.locator('#vision-task').selectOption('person')
 await page.getByRole('button',{name:'Start local vision'}).click()
 await page.waitForFunction(()=>document.querySelector('[role=status]')?.textContent==='Running locally · camera active',null,{timeout:90000})
 await page.locator('canvas[aria-label="Person foreground confidence mask"]').waitFor({timeout:30000})
 await page.getByRole('button',{name:'Stop and release camera'}).click()
 await page.waitForFunction(()=>document.querySelector('[role=status]')?.textContent==='Camera off',null,{timeout:20000})

 if(pageErrors.length)throw new Error('Page exceptions: '+pageErrors.join('; '))
 if(requestErrors.some(err=>err.includes('/ort/')))throw new Error('Local ONNX runtime failed: '+requestErrors.join('; '))
 console.log('PASS: real SHA-verified selfie model and pinned Q8 depth inference, nonconstant grayscale pixels, local ONNX runtime, existing camera stop')
}finally{
 await browser?.close()
 try{process.kill(-server.pid,'SIGTERM')}catch{}
}
