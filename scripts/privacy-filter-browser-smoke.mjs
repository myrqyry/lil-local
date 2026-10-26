import { chromium } from 'playwright'
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_BIN??'/usr/bin/chromium',headless:true,args:['--enable-unsafe-webgpu']})
try {
 const page=await browser.newPage()
 page.on('console',m=>{const t=m.text();if(t.startsWith('PRIVACY_SMOKE')||m.type()==='error')console.log('BROWSER',t.slice(0,850))})
 page.on('pageerror',e=>console.log('PAGE_ERROR',String(e).slice(0,850)))
 await page.goto('http://127.0.0.1:4391/',{waitUntil:'networkidle'})
 console.log('GPU',await page.evaluate(async()=>{
   const a=await navigator.gpu?.requestAdapter()
   return {webgpu:!!navigator.gpu,adapter:!!a,f16:a?.features.has('shader-f16')??false}
 }))
 await page.click('#run')
 await page.waitForFunction(()=>window.privacyResult!==undefined,null,{timeout:300000})
 const result=await page.evaluate(()=>window.privacyResult)
 console.log('FINAL',JSON.stringify(result))
 if(!result?.ok)process.exitCode=1
}finally{await browser.close()}
