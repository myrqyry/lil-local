import { chromium } from 'playwright'
import { writeFile } from 'node:fs/promises'
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_BIN ?? '/usr/bin/chromium',
  headless: true,
  args: ['--enable-unsafe-webgpu'],
})
try {
  const page = await browser.newPage()
  page.on('console', msg => {
    const s=msg.text()
    if(s.startsWith('PHONON_SMOKE') || msg.type()==='error') console.log('PAGE',s.slice(0,1000))
  })
  page.on('pageerror',e=>console.log('PAGE_ERROR',String(e).slice(0,900)))
  await page.goto('http://127.0.0.1:4388/',{waitUntil:'networkidle'})
  await page.locator('#run').waitFor()
  console.log('BROWSER',await page.evaluate(async()=>({
    webgpu:!!navigator.gpu,
    adapter:await navigator.gpu?.requestAdapter().then(a=>a?.info?.description??a?.info?.vendor??'not available').catch(e=>String(e)),
  })))
  await page.click('#run')
  try {
    await page.waitForFunction(()=>window.phononResult!==undefined,null,{timeout:180000})
  } catch(e) { console.log('SMOKE_TIMEOUT',String(e)); console.log('TEXT',await page.locator('#log').textContent()); throw e }
  const result=await page.evaluate(()=>window.phononResult)
  console.log('RESULT',JSON.stringify(result))
  if(!result?.ok) throw new Error('Phonon model inference failed')
  const b64=await page.evaluate(async()=>{
    const url=(document.querySelector('#audio')).src
    const bytes=new Uint8Array(await (await fetch(url)).arrayBuffer())
    let binary=''
    for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192))
    return btoa(binary)
  })
  await writeFile('/tmp/lil-local-phonon-browser.wav',Buffer.from(b64,'base64'))
  console.log('WAVE_WRITTEN /tmp/lil-local-phonon-browser.wav')
} finally {await browser.close()}
