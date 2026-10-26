import { BrowserSubstrateRunner } from '../../packages/inference-core/src/index'
import { createPrivacyFilterSubstrate, redactPrivacySpans } from '../../packages/privacy-filter/src/index'
declare global {interface Window { privacyResult?: unknown }}
const log=(message:string)=>{console.log('PRIVACY_SMOKE',message);(document.getElementById('log') as HTMLElement).textContent += message+'\n'}
async function run(){
  const original='My name is Alice Smith and my email address is alice@example.com.'
  log('adapter-check')
  const adapter=await navigator.gpu?.requestAdapter()
  log('shader-f16='+(adapter?.features.has('shader-f16')??false))
  const runner=new BrowserSubstrateRunner(createPrivacyFilterSubstrate({device:'webgpu',onProgress:p=>{
    if(p.status==='initiate'||p.status==='ready'||p.status==='done') log('asset '+String(p.file??'')+' '+String(p.status))
  }}))
  try{
    log('loading')
    await runner.load()
    log('loaded')
    const result=await runner.run({text:original})
    const redacted=redactPrivacySpans(original,result.spans)
    const summary={ok:true,spans:result.spans,redacted,device:result.device,dtype:result.dtype}
    window.privacyResult=summary
    log('SUCCESS '+JSON.stringify(summary))
  }catch(e){
    const summary={ok:false,error:String(e)}
    window.privacyResult=summary
    log('FAILED '+String(e))
    console.error(e)
  }finally{await runner.dispose()}
}
(document.querySelector('#run') as HTMLButtonElement).addEventListener('click',()=>void run())
