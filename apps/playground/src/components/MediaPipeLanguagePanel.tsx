import {useEffect,useRef,useState} from 'react'
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {
  createLanguageDetectorSubstrate,
  type LanguageDetectionResult,
  type TextLoadProgress,
} from '@lil-local/text-mediapipe'

type Runner=BrowserSubstrateRunner<string,LanguageDetectionResult,TextLoadProgress>
const EXAMPLES=[
  {label:'English',text:'This quiet little library is a wonderful place to read a book and spend the afternoon.'},
  {label:'Español',text:'Esta biblioteca es un lugar tranquilo donde podemos leer libros y pasar toda la tarde.'},
  {label:'日本語',text:'今日はとても天気がいいので、近くの公園で本を読んだり散歩したりしたいです。'},
]

export function MediaPipeLanguagePanel(){
  const runnerRef=useRef<Runner|null>(null)
  const abortRef=useRef<AbortController|null>(null)
  const mounted=useRef(true)
  const [text,setText]=useState(EXAMPLES[0].text)
  const [delegate,setDelegate]=useState<'GPU'|'CPU'>('GPU')
  const [ready,setReady]=useState(false)
  const [busy,setBusy]=useState(false)
  const [status,setStatus]=useState('Model not loaded; no network request has been made.')
  const [error,setError]=useState('')
  const [result,setResult]=useState<LanguageDetectionResult|null>(null)
  useEffect(()=>{
    mounted.current=true
    return ()=>{
      mounted.current=false
      abortRef.current?.abort()
      const instance=runnerRef.current
      runnerRef.current=null
      if(instance)void instance.dispose()
    }
  },[])

  const load=async()=>{
    if(busy||ready)return
    setBusy(true);setError('');setResult(null)
    const controller=new AbortController()
    abortRef.current=controller
    const runner=new BrowserSubstrateRunner(createLanguageDetectorSubstrate({
      delegate,maxResults:5,scoreThreshold:0,
    }))
    runnerRef.current=runner
    try{
      await runner.load(stage=>{
        if(mounted.current&&!controller.signal.aborted){
          setStatus({
            model:'Downloading official 315 KB model…',
            integrity:'Verifying complete model SHA-256…',
            wasm:'Initializing MediaPipe Tasks Text '+(delegate==='GPU'?'WebGL':'CPU')+' runtime…',
          }[stage])
        }
      },controller.signal)
      if(!mounted.current||controller.signal.aborted)return
      setReady(true)
      setStatus('Local language detector ready · '+(delegate==='GPU'?'MediaPipe WebGL':'CPU/WASM'))
    }catch(cause){
      if(mounted.current&&!controller.signal.aborted){
        setError(cause instanceof Error?cause.message:String(cause))
        setStatus('Detector initialization failed; no fallback was attempted.')
      }
      if(runnerRef.current===runner){
        runnerRef.current=null
        await runner.dispose().catch(()=>undefined)
      }
    }finally{
      if(abortRef.current===controller)abortRef.current=null
      if(mounted.current)setBusy(false)
    }
  }

  const detect=async()=>{
    if(!ready||busy||!runnerRef.current)return
    setBusy(true);setError('')
    try{
      const prediction=await runnerRef.current.run(text)
      if(!mounted.current)return
      setResult(prediction)
      setStatus('Text classified locally. '+prediction.languages.length+' language candidates.')
    }catch(cause){
      if(mounted.current){
        setError(cause instanceof Error?cause.message:String(cause))
        setStatus('Language classification failed.')
      }
    }finally{if(mounted.current)setBusy(false)}
  }

  const unload=async()=>{
    if(!ready||busy)return
    setBusy(true)
    const runner=runnerRef.current
    runnerRef.current=null
    try{await runner?.dispose()}
    catch(cause){if(mounted.current)setError(String(cause))}
    finally{
      if(mounted.current){
        setReady(false);setResult(null);setBusy(false)
        setStatus('Model unloaded.')
      }
    }
  }

  return <section className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4 text-on-surface">
    <p className="text-sm text-on-surface-variant">
      Detect the language of a whole text passage using a verified 315 KB
      MediaPipe model. No translation, transcription, or automatic voice
      changes. Short, mixed-language, and ambiguous text may be uncertain.
    </p>
    <div className="flex flex-wrap gap-2">
      {EXAMPLES.map(example=><button key={example.label} type="button"
        className="rounded-lg border border-outline-variant px-3 py-2 text-sm"
        disabled={busy}
        onClick={()=>{setText(example.text);setResult(null)}}>
        {example.label}
      </button>)}
    </div>
    <label className="block text-sm font-semibold">Text to classify
      <textarea aria-label="Text for local language identification" value={text}
        onChange={e=>{setText(e.target.value);setResult(null)}}
        rows={4} maxLength={32768} disabled={busy}
        className="mt-2 w-full rounded-lg border border-outline-variant bg-surface-container px-3 py-2 font-normal"/>
    </label>
    <label className="block text-sm">Runtime
      <select aria-label="Language detector delegate"
        disabled={ready||busy} value={delegate}
        onChange={e=>setDelegate(e.target.value as 'GPU'|'CPU')}
        className="ml-2 rounded-lg border border-outline-variant bg-surface-container px-2 py-2">
        <option value="GPU">GPU (MediaPipe WebGL, not WebGPU)</option>
        <option value="CPU">CPU / WASM</option>
      </select>
    </label>
    <div className="flex flex-wrap gap-2">
      <button type="button" disabled={busy||ready} onClick={()=>void load()}
        className="rounded-lg bg-primary px-3 py-2 font-medium text-on-primary disabled:opacity-50">
        Load language detector
      </button>
      <button type="button" disabled={busy||!ready||!text.trim()}
        onClick={()=>void detect()}
        className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
        Detect language
      </button>
      <button type="button" disabled={busy||!ready}
        onClick={()=>void unload()}
        className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
        Unload
      </button>
    </div>
    <p role="status" className="text-sm text-on-surface-variant">{status}</p>
    {error && <p role="alert" className="text-sm text-error">{error}</p>}
    {result&&<div aria-label="Language prediction results" className="space-y-2">
      {result.languages.length===0
        ?<p className="text-sm">No confident language returned by the model.</p>
        :result.languages.map((prediction,i)=><div key={prediction.languageCode+'-'+i}
          className="rounded-lg border border-outline-variant p-3">
          <div className="flex justify-between gap-2 text-sm">
            <strong>{prediction.languageCode}</strong>
            <span>{(prediction.probability*100).toFixed(1)}%</span>
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-container">
            <div className="h-full bg-primary" style={{width:(100*prediction.probability)+'%'}}/>
          </div>
        </div>)}
    </div>}
    <p className="text-xs text-on-surface-variant">
      Model and matching WASM are downloaded on explicit load, unless
      provided locally. This is inference on the full passage—not
      identification of each language span. MediaPipe SDK telemetry
      may occur; no text is intentionally uploaded by this adapter.
    </p>
  </section>
}
