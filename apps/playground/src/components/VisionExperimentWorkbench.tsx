import {useEffect, useRef, useState} from 'react'
import {
  MAX_EXPERIMENTS, renderSampledPixels,
  type SampledField, type VisionExperiment,
} from './visionExperiments'

interface Props {
  entries: readonly VisionExperiment[]
  onClear: () => void
}

/** A read-only numerical view; no model execution or image uploads. */
function FieldCanvas({
  raster, mode, threshold, label, className = '',
}: {
  raster: SampledField
  mode: 'raw'|'cutoff'
  threshold: number
  label: string
  className?: string
}) {
  const canvasRef=useRef<HTMLCanvasElement>(null)
  useEffect(()=>{
    const canvas=canvasRef.current
    if(!canvas) return
    const ctx=canvas.getContext('2d')
    if(!ctx) return
    canvas.width=raster.width
    canvas.height=raster.height
    const pixels = new Uint8ClampedArray(new ArrayBuffer(raster.values.length * 4))
    pixels.set(renderSampledPixels(raster,{mode,threshold}))
    ctx.putImageData(new ImageData(pixels,raster.width,raster.height),0,0)
  },[raster,mode,threshold])
  return <canvas ref={canvasRef} aria-label={label}
    className={'h-full w-full bg-black object-contain '+className}/>
}

function RunImage({entry,mode,threshold}:{
  entry:VisionExperiment
  mode:'raw'|'cutoff'
  threshold:number
}) {
  if(!entry.raster) return <div className="flex h-full min-h-40 items-center justify-center bg-surface-container-high p-4 text-center text-xs text-on-surface-variant">
    Typed result recorded · no raster image produced for this task
  </div>
  return <FieldCanvas raster={entry.raster} mode={mode} threshold={threshold}
    label={'Output of run '+entry.id}/>
}

const number=(value:number)=>Number.isFinite(value)?value.toFixed(1):'–'
function RunStats({entry}: {entry:VisionExperiment}) {
  return <div className="space-y-1 text-xs text-on-surface-variant">
    <div className="font-semibold text-on-surface">#{entry.id} · {entry.task} <span className="font-normal">· {entry.summary}</span></div>
    <div className="font-mono">Load {number(entry.loadMs)} ms · Inference {number(entry.inferenceMs)} ms</div>
    <div>Requested: {entry.requestedBackend} · Source {entry.image.width}×{entry.image.height}</div>
    {entry.raster && <div className="font-mono">
      Sampled floats {entry.raster.width}×{entry.raster.height} (from {entry.raster.sourceWidth}×{entry.raster.sourceHeight})
      {' · '}min {entry.raster.minimum.toFixed(3)} · max {entry.raster.maximum.toFixed(3)} · avg {entry.raster.mean.toFixed(3)}
    </div>}
  </div>
}

export function VisionExperimentWorkbench({entries,onClear}: Props) {
  const [firstId,setFirstId]=useState<number|null>(null)
  const [secondId,setSecondId]=useState<number|null>(null)
  const [split,setSplit]=useState(50)
  const [mode,setMode]=useState<'raw'|'cutoff'>('raw')
  const [threshold,setThreshold]=useState(0.5)
  const [confirmClear,setConfirmClear]=useState(false)
  const first=entries.find(e=>e.id===firstId)??entries[Math.min(1,entries.length-1)]
  const second=entries.find(e=>e.id===secondId)??entries[0]
  const comparable=entries.length>1 && first && second && first.id!==second.id
  const bothRaster=Boolean(comparable && first.raster && second.raster)
  const sameSource=Boolean(bothRaster && first.image.preview===second.image.preview)
  const canCutoff=Boolean(first?.raster?.kind==='person-mask'||second?.raster?.kind==='person-mask')
  const visibleRuns=first&&second ? entries.length>1 ? [first,second] : [second] : []

  return <section aria-label="Still-image experiment workbench"
    className="space-y-4 rounded-2xl border border-outline-variant bg-surface p-4">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 className="text-base font-semibold text-on-surface">Experiment workbench</h2>
        <p className="mt-1 text-xs text-on-surface-variant">
          Last {MAX_EXPERIMENTS} successful still-image runs · session memory only · no automatic downloads
        </p>
      </div>
      {entries.length>0 && (
        <button type="button"
          className="rounded-lg border border-outline-variant px-3 py-1.5 text-xs text-on-surface"
          onClick={()=>{
            if(!confirmClear){setConfirmClear(true);return}
            onClear();setConfirmClear(false);setFirstId(null);setSecondId(null)
          }}>{confirmClear?'Confirm clear runs':'Clear run history'}</button>
      )}
    </div>
    {entries.length===0
      ? <p className="rounded-xl border border-dashed border-outline-variant p-4 text-sm text-on-surface-variant">
        Choose a still image and click <strong>Analyze selected image locally</strong>.
        Successful model results and their actual timings will appear here. Live camera frames are not recorded.
      </p>
      : <>
        {entries.length>1 ? <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-xs text-on-surface-variant">A · earlier or selected run
            <select aria-label="Experiment A" value={first?.id??''}
              onChange={event=>setFirstId(Number(event.target.value))}
              className="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container px-2 py-2 text-on-surface">
              {entries.map(entry=><option key={entry.id} value={entry.id}>#{entry.id} {entry.task} · {entry.image.name}</option>)}
            </select>
          </label>
          <label className="text-xs text-on-surface-variant">B · latest or selected run
            <select aria-label="Experiment B" value={second?.id??''}
              onChange={event=>setSecondId(Number(event.target.value))}
              className="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container px-2 py-2 text-on-surface">
              {entries.map(entry=><option key={entry.id} value={entry.id}>#{entry.id} {entry.task} · {entry.image.name}</option>)}
            </select>
          </label>
        </div> : <p className="text-xs text-on-surface-variant">One run captured. Analyze another still image to compare actual model outputs.</p>}

        {canCutoff && <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs text-on-surface-variant">Foreground display
            <select aria-label="Foreground display mode" value={mode}
              onChange={event=>setMode(event.target.value as 'raw'|'cutoff')}
              className="ml-2 rounded-lg border border-outline-variant bg-surface-container px-2 py-1 text-on-surface">
              <option value="raw">Raw confidence (grayscale)</option>
              <option value="cutoff">Binary cutoff</option>
            </select>
          </label>
          {mode==='cutoff' && <label className="flex items-center gap-2 text-xs text-on-surface-variant">
            Confidence ≥ {threshold.toFixed(2)}
            <input aria-label="Foreground confidence threshold" type="range"
              min="0" max="1" step="0.01" value={threshold}
              onChange={event=>setThreshold(Number(event.target.value))}/>
          </label>}
        </div>}

        {first && second && <div className="grid gap-3 sm:grid-cols-2">
          {visibleRuns.map((entry,i)=><div key={i+'-'+entry.id} className="min-w-0 space-y-2">
            <div className="rounded-xl border border-outline-variant p-2">
              <p className="mb-2 text-xs font-semibold text-on-surface">Input {i===0?'A':'B'} · {entry.image.name}</p>
              <img alt={'Source image of run '+entry.id} src={entry.image.preview}
                className="aspect-[4/3] w-full rounded-lg bg-black object-contain"/>
            </div>
            <RunStats entry={entry}/>
          </div>)}
        </div>}

        {first && second && <div className="space-y-2">
          <p className="text-sm font-semibold text-on-surface">Model output</p>
          {bothRaster && sameSource
            ? <div className="space-y-2">
              <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-black">
                <div className="absolute inset-0"><RunImage entry={second} mode={mode} threshold={threshold}/></div>
                <div className="absolute inset-0" style={{clipPath:`inset(0 ${100-split}% 0 0)`}}>
                  <RunImage entry={first} mode={mode} threshold={threshold}/>
                </div>
                <div className="pointer-events-none absolute inset-y-0 w-px bg-white"
                  style={{left:split+'%'}}/>
                <span className="absolute left-2 top-2 rounded bg-black/70 px-2 py-1 text-xs text-white">A</span>
                <span className="absolute right-2 top-2 rounded bg-black/70 px-2 py-1 text-xs text-white">B</span>
              </div>
              <label className="flex items-center gap-3 text-xs text-on-surface-variant">A/B comparison split
                <input aria-label="A/B comparison split" className="min-w-0 flex-1" type="range" min="0" max="100"
                  value={split} onChange={event=>setSplit(Number(event.target.value))}/>
                <span className="font-mono">{split}%</span>
              </label>
            </div>
            : <div className="grid gap-3 sm:grid-cols-2">
              {visibleRuns.map((entry,i)=><div key={i+'-'+entry.id} className="space-y-1">
                <p className="text-xs text-on-surface-variant">Output {i===0?'A':'B'}</p>
                <div className="aspect-[4/3] overflow-hidden rounded-xl">
                  <RunImage entry={entry} mode={mode} threshold={threshold}/>
                </div>
              </div>)}
            </div>}
          {!sameSource && bothRaster && <p className="text-xs text-on-surface-variant">
            Different source images: side-by-side only; pixel-aligned split comparison is disabled.
          </p>}
        </div>}

        <p className="text-xs text-on-surface-variant">
          Input previews show the selected image, <strong>not</strong> the SDK's internal model tensor.
          Raster values are bounded nearest-neighbor samples of returned float fields;
          cutoff/grayscale changes never rerun inference. A/B is visual inspection,
          not a metric of model accuracy. No results are stored after leaving this page.
        </p>
        <div className="space-y-1" aria-label="Experiment run ledger">
          <p className="text-xs font-semibold text-on-surface">Recent runs</p>
          {entries.map(entry=><div key={entry.id}
            className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-outline-variant px-3 py-2 text-xs">
            <span>#{entry.id} · {entry.task} · {entry.image.name}</span>
            <span className="font-mono text-on-surface-variant">
              {number(entry.loadMs)} ms load / {number(entry.inferenceMs)} ms infer
            </span>
          </div>)}
        </div>
      </>}
  </section>
}
