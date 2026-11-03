import { PhononPipeline } from '../../packages/phonon/src/index'
import { PHONON_HF_BASE } from '../../packages/phonon/src/manifest'
import type { RuntimeContext } from '../../packages/inference-core/src/index'
const logEl = document.querySelector('#log') as HTMLPreElement
const say = (message: string) => { logEl.textContent += message + '\n'; console.log('PHONON_SMOKE', message) }
function wave(samples: Float32Array, rate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const bytes = new Uint8Array(buffer), view = new DataView(buffer)
  const tag = (offset: number, value: string) => { for (let i = 0; i < value.length; i++) bytes[offset + i] = value.charCodeAt(i) }
  tag(0, 'RIFF');view.setUint32(4,buffer.byteLength-8,true)
  tag(8,'WAVE');tag(12,'fmt ');view.setUint32(16,16,true)
  view.setUint16(20,1,true);view.setUint16(22,1,true)
  view.setUint32(24,rate,true);view.setUint32(28,rate*2,true)
  view.setUint16(32,2,true);view.setUint16(34,16,true)
  tag(36,'data');view.setUint32(40,samples.length*2,true)
  for(let i=0;i<samples.length;i++)view.setInt16(44+i*2,Math.round(Math.max(-1,Math.min(1,samples[i]))*32767),true)
  return new Blob([buffer],{type:'audio/wav'})
}
declare global { interface Window { phononResult?: { ok: boolean; message?: string; peak?: number; rms?: number; samples?: number; backend?: string } } }
async function run() {
  logEl.textContent = ''
  const useGPU = !!navigator.gpu
  const backend = useGPU ? 'webgpu' : 'wasm'
  say('backend=' + backend)
  const pipeline = new PhononPipeline()
  pipeline.onProgress = progress => say('loading ' + progress.step + '/' + progress.total)
  try {
    await pipeline.load({
      backend,
      assets: { async resolve(asset) {
        const response = await fetch(PHONON_HF_BASE + asset.path)
        if (!response.ok) throw new Error('Fetch '+asset.path+' returned '+response.status)
        return await response.arrayBuffer()
      } },
      liteRt: {},
    } as RuntimeContext)
    say('loaded')
    const result = await pipeline.run({text:'Hello from Lil Local. This is a real Phonon speech test.'},{temperature:0})
    const peak = result.samples.reduce((m,s)=>Math.max(m,Math.abs(s)),0)
    const rms = Math.sqrt(result.samples.reduce((m,s)=>m+s*s,0)/result.samples.length)
    const url = URL.createObjectURL(wave(result.samples,result.sampleRate))
    const audio = document.querySelector('#audio') as HTMLAudioElement
    audio.src = url
    window.phononResult = {ok:true,peak,rms,samples:result.samples.length,backend}
    say('SUCCESS '+JSON.stringify(window.phononResult))
  } catch(err) {
    say('FAILED '+String(err))
    window.phononResult = {ok:false,message:String(err),backend}
    console.error(err)
  } finally {
    try {await pipeline.dispose()} catch(e){say('DISPOSE_FAILED '+String(e))}
  }
}
(document.querySelector('#run') as HTMLButtonElement).addEventListener('click',()=>{void run()})
