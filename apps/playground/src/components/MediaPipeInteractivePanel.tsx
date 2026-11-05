import {useEffect, useRef, useState, type PointerEvent} from 'react'
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {
  createInteractiveSegmenterSubstrate,
  type InteractiveSelectionRequest,
  type InteractiveSelectionResult,
  type InteractiveSelectionStroke,
  type VisionProgress,
} from '@lil-local/vision-mediapipe'

type Runner=BrowserSubstrateRunner<InteractiveSelectionRequest,InteractiveSelectionResult,VisionProgress>
type Mask=Extract<InteractiveSelectionResult,{kind:'interactive-segmentation'}>
type Point={x:number;y:number}

function normalizedPoint(event:PointerEvent<SVGSVGElement>):Point{
  const rect=event.currentTarget.getBoundingClientRect()
  return {
    x:Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),
    y:Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height)),
  }
}
function strokeLength(points:readonly Point[]):number{
  return points.slice(1).reduce((total,point,i)=>total+Math.hypot(
    point.x-points[i].x,point.y-points[i].y),0)
}
function strokePath(points:readonly Point[]):string{
  return points.map((point,i)=>(i?'L':'M')+point.x+' '+point.y).join(' ')
}

export function MediaPipeInteractivePanel(){
  const mounted=useRef(true)
  const runnerRef=useRef<Runner|null>(null)
  const abortRef=useRef<AbortController|null>(null)
  const drawingRef=useRef<InteractiveSelectionStroke|null>(null)
  const strokesRef=useRef<InteractiveSelectionStroke[]>([])
  const canvasRef=useRef<HTMLCanvasElement>(null)
  const [imageFile,setImageFile]=useState<File|null>(null)
  const [preview,setPreview]=useState<string|null>(null)
  const [ready,setReady]=useState(false)
  const [busy,setBusy]=useState(false)
  const [mode,setMode]=useState<InteractiveSelectionStroke['mode']>('positive')
  const [strokes,setStrokes]=useState<InteractiveSelectionStroke[]>([])
  const [drawing,setDrawing]=useState<InteractiveSelectionStroke|null>(null)
  const [mask,setMask]=useState<Mask|null>(null)
  const [status,setStatus]=useState('Select an image, then explicitly load the model.')
  const [error,setError]=useState('')

  useEffect(()=>{
    mounted.current=true
    return ()=>{
      mounted.current=false
      abortRef.current?.abort()
      const runner=runnerRef.current
      runnerRef.current=null
      if(runner)void runner.dispose()
    }
  },[])
  useEffect(()=>{
    if(!imageFile){setPreview(null);return}
    const url=URL.createObjectURL(imageFile)
    setPreview(url)
    return ()=>URL.revokeObjectURL(url)
  },[imageFile])
  useEffect(()=>{
    if(!mask||!canvasRef.current)return
    const canvas=canvasRef.current
    canvas.width=mask.width
    canvas.height=mask.height
    const ctx=canvas.getContext('2d')
    if(!ctx)return
    const pixels=new Uint8ClampedArray(mask.width*mask.height*4)
    for(let i=0;i<mask.confidence.length;i++){
      const opacity=mask.confidence[i]
      pixels[4*i]=204
      pixels[4*i+1]=146
      pixels[4*i+2]=96
      pixels[4*i+3]=Math.round(opacity*170)
    }
    ctx.putImageData(new ImageData(pixels,mask.width,mask.height),0,0)
  },[mask])

  const clearDrawing=()=>{
    drawingRef.current=null
    setDrawing(null)
    strokesRef.current=[]
    setStrokes([])
    setMask(null)
  }

  const load=async()=>{
    if(!imageFile||busy||ready)return
    setBusy(true)
    setError('')
    clearDrawing()
    const controller=new AbortController()
    abortRef.current=controller
    const runner=new BrowserSubstrateRunner(createInteractiveSegmenterSubstrate())
    runnerRef.current=runner
    let bitmap:ImageBitmap|null=null
    try{
      if(!imageFile.type.startsWith('image/'))throw new Error('Select an image file')
      setStatus('Loading SHA-256-verified MagicTouch V2 model…')
      await runner.load(stage=>{
        if(mounted.current&&!controller.signal.aborted)setStatus({
          model:'Downloading official 30.5 MB model…',
          integrity:'Verifying complete SHA-256…',
          wasm:'Initializing local MediaPipe CPU/WASM…',
        }[stage])
      },controller.signal)
      if(!mounted.current||controller.signal.aborted)return
      bitmap=await createImageBitmap(imageFile)
      if(!mounted.current||controller.signal.aborted)return
      setStatus('Encoding image once for repeated refinement…')
      await runner.run({kind:'set-image',image:bitmap})
      if(!mounted.current||controller.signal.aborted)return
      setReady(true)
      setStatus('Image encoded. Drag a positive stroke over the part to select.')
    }catch(cause){
      if(mounted.current&&!controller.signal.aborted){
        setError(cause instanceof Error?cause.message:String(cause))
        setStatus('MagicTouch model unavailable')
      }
      if(runnerRef.current===runner){
        runnerRef.current=null
        await runner.dispose().catch(()=>undefined)
      }
    }finally{
      bitmap?.close()
      if(abortRef.current===controller)abortRef.current=null
      if(mounted.current)setBusy(false)
    }
  }

  const runStrokes=async(next:InteractiveSelectionStroke[])=>{
    const runner=runnerRef.current
    if(!runner||!ready||busy||!next.length)return
    setBusy(true);setError('')
    try{
      setStatus('Refining selection from cached image encoding…')
      const result=await runner.run({kind:'segment',strokes:next})
      if(!mounted.current)return
      if(result.kind!=='interactive-segmentation')throw new Error('Unexpected model output')
      setMask(result)
      setStatus('Refined selection locally · '+next.length+' stroke'+(next.length===1?'':'s'))
    }catch(cause){
      if(mounted.current){
        setError(cause instanceof Error?cause.message:String(cause))
        setStatus('Segmentation failed')
      }
    }finally{if(mounted.current)setBusy(false)}
  }

  const unload=async()=>{
    if(busy)return
    setBusy(true)
    const runner=runnerRef.current
    runnerRef.current=null
    try{await runner?.dispose()}
    catch(cause){if(mounted.current)setError(String(cause))}
    finally{
      if(mounted.current){
        clearDrawing()
        setReady(false);setBusy(false)
        setStatus('Model unloaded. Select a file and load when ready.')
      }
    }
  }

  const onDown=(event:PointerEvent<SVGSVGElement>)=>{
    if(!ready||busy||strokesRef.current.length>=32||event.button!==0)return
    event.currentTarget.setPointerCapture(event.pointerId)
    const point=normalizedPoint(event)
    const next:InteractiveSelectionStroke={mode,points:[point]}
    drawingRef.current=next
    setDrawing(next)
  }
  const onMove=(event:PointerEvent<SVGSVGElement>)=>{
    const current=drawingRef.current
    if(!current||busy)return
    const next=normalizedPoint(event)
    const previous=current.points[current.points.length-1]
    if(Math.hypot(next.x-previous.x,next.y-previous.y)<.003)return
    const updated={...current,points:[...current.points,next]}
    drawingRef.current=updated
    setDrawing(updated)
  }
  const onUp=(event:PointerEvent<SVGSVGElement>)=>{
    const current=drawingRef.current
    if(!current)return
    if(event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId)
    drawingRef.current=null
    setDrawing(null)
    if(current.points.length<2||strokeLength(current.points)<.05){
      setStatus('Stroke too short. Drag at least 5% of the image width.')
      return
    }
    const next=[...strokesRef.current,current]
    strokesRef.current=next
    setStrokes(next)
    void runStrokes(next)
  }
  const removeLast=()=>{
    if(!ready||busy||!strokes.length)return
    const next=strokesRef.current.slice(0,-1)
    strokesRef.current=next
    setStrokes(next)
    if(next.length)void runStrokes(next)
    else{setMask(null);setStatus('Strokes cleared; image encoding is still cached.')}
  }
  const selected=mask?.confidence.reduce((count,value)=>count+(value>.5?1:0),0)??0

  return <section className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4 text-on-surface">
    <p className="text-sm text-on-surface-variant">
      MagicTouch V2 reuses a single encoded image for positive, negative and lasso
      strokes. Unlike a SAM request, each refinement does not rerun the image
      encoder. Everything runs locally after an explicit model download.
    </p>
    <label className="block text-sm font-semibold">Local image (no camera)
      <input type="file" accept="image/*" aria-label="MagicTouch input image"
        disabled={ready||busy} className="mt-2 block w-full text-sm"
        onChange={event=>{setImageFile(event.target.files?.[0]??null);clearDrawing()}}/>
    </label>
    <div className="flex flex-wrap gap-2">
      <button type="button" disabled={!imageFile||ready||busy} onClick={()=>void load()}
        className="rounded-lg bg-primary px-3 py-2 text-on-primary disabled:opacity-50">
        {busy&&!ready?'Encoding…':ready?'Image ready':'Load & encode image'}
      </button>
      <button type="button" disabled={busy||!ready} onClick={()=>void unload()}
        className="rounded-lg border border-outline-variant px-3 py-2 disabled:opacity-50">
        Unload model
      </button>
    </div>
    <p role="status" className="text-sm text-on-surface-variant">{status}</p>
    {error&&<p role="alert" className="text-sm text-error">{error}</p>}
    <div className="flex flex-wrap items-center gap-3">
      <label className="text-sm">Brush mode
        <select aria-label="MagicTouch brush mode" disabled={!ready||busy}
          value={mode} onChange={event=>setMode(event.target.value as InteractiveSelectionStroke['mode'])}
          className="ml-2 rounded-lg border border-outline-variant bg-surface-container px-2 py-2">
          <option value="positive">Include</option>
          <option value="negative">Exclude</option>
          <option value="lasso">Lasso</option>
        </select>
      </label>
      <button type="button" disabled={!ready||busy||!strokes.length} onClick={removeLast}
        className="rounded-lg border border-outline-variant px-3 py-2 text-sm disabled:opacity-50">
        Undo stroke
      </button>
      <button type="button" disabled={!ready||busy||!strokes.length}
        onClick={()=>{clearDrawing();setStatus('Strokes cleared. Image encoding retained.')}}
        className="rounded-lg border border-outline-variant px-3 py-2 text-sm disabled:opacity-50">
        Clear strokes
      </button>
    </div>
    {preview&&<div className="relative overflow-hidden rounded-xl" aria-label="MagicTouch image workbench">
      <img alt="Selected image for interactive segmentation" src={preview} className="block w-full"/>
      {mask&&<canvas ref={canvasRef} aria-label="MagicTouch confidence overlay"
        className="pointer-events-none absolute inset-0 h-full w-full"/>}
      <svg viewBox="0 0 1 1" preserveAspectRatio="none"
        role="img" aria-label="Draw MagicTouch strokes over image"
        onPointerDown={onDown} onPointerMove={onMove}
        onPointerUp={onUp} onPointerCancel={()=>{
          drawingRef.current=null;setDrawing(null)
        }}
        className="absolute inset-0 h-full w-full"
        style={{touchAction:'none',cursor:ready&&!busy?'crosshair':'default',pointerEvents:ready&&!busy?'auto':'none'}}>
        {[...strokes,...(drawing?[drawing]:[])].map((stroke,i)=>
          <path key={i} d={strokePath(stroke.points)} fill="none"
            stroke={stroke.mode==='negative'?'#85b5bb':stroke.mode==='lasso'?'#94ae89':'#e9b16b'}
            strokeWidth={0.007} strokeLinecap="round" strokeLinejoin="round"/>)}
      </svg>
    </div>}
    <p aria-label="MagicTouch mask status" className="text-sm text-on-surface-variant">
      {strokes.length} strokes · {mask
        ?mask.width+' × '+mask.height+' confidence mask · '+(100*selected/(mask.width*mask.height)).toFixed(1)+'% selected above 0.5'
        :'No mask yet'}
    </p>
    <p className="text-xs text-on-surface-variant">
      Drag over the image (a click alone is too short), then refine with Exclude
      or Lasso. Undo and Clear reuse the encoded image. This is soft confidence
      masking, not a semantic category map or native transparent cutout.
      CPU/WASM inference may take several seconds per stroke. Large photos
      can consume substantial native memory. The initial model and runtime
      downloads may require a network connection.
    </p>
  </section>
}
