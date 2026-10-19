import { useEffect, useRef, useState } from 'react'
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import {
  D1_MODEL_REVISION,
  type D1Decision,
  type D1Question,
  type D1LoadPhase,
} from '@lil-local/decision-onnx'
import { d1OnnxSubstrate, type D1SubstrateInput } from '../substrates/d1Onnx'

const defaultState = 'I was charged twice for the same order. I need a refund.'
const defaultOptions = 'billing: Charges, refunds, invoices\ntechnical: App or website faults\nfraud: Suspected unauthorized use'

function parseChoiceOptions(raw: string): Record<string, string> {
  const criteria: Record<string, string> = {}
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue
    const separator = line.indexOf(':')
    const label = (separator < 0 ? line : line.slice(0, separator)).trim()
    const description = separator < 0 ? '' : line.slice(separator + 1).trim()
    if (!label || Object.prototype.hasOwnProperty.call(criteria, label)) {
      throw new Error('Options need unique nonempty names')
    }
    criteria[label] = description
  }
  return criteria
}

export function D1DecisionPanel() {
  const [state, setState] = useState(defaultState)
  const [question, setQuestion] = useState('Which team should handle this?')
  const [options, setOptions] = useState(defaultOptions)
  const [kind, setKind] = useState<'choice' | 'noul'>('choice')
  const [status, setStatus] = useState('Not loaded')
  const [error, setError] = useState('')
  const [answer, setAnswer] = useState<D1Decision | null>(null)
  const [loading, setLoading] = useState(false)
  const [deciding, setDeciding] = useState(false)
  const pipeline = useRef<BrowserSubstrateRunner<D1SubstrateInput, D1Decision, D1LoadPhase> | null>(null)
  const pendingLoad = useRef<BrowserSubstrateRunner<D1SubstrateInput, D1Decision, D1LoadPhase> | null>(null)
  const loadAbort = useRef<AbortController | null>(null)
  const [stopping, setStopping] = useState(false)
  const generation = useRef(0)

  useEffect(() => () => {
    generation.current += 1
    loadAbort.current?.abort()
    const pending = pendingLoad.current
    pendingLoad.current = null
    loadAbort.current = null
    if (pending) void pending.dispose().catch(error => console.error('Pending D1 release failed:', error))
    const instance = pipeline.current
    pipeline.current = null
    if (instance) void instance.dispose().catch(error => console.error('D1 release failed:', error))
  }, [])

  async function load() {
    if (loading || pipeline.current) return
    const requestId = ++generation.current
    setLoading(true)
    setError('')
    setAnswer(null)
    setStatus('Starting explicit WebGPU load…')
    const controller = new AbortController()
    const loaded = new BrowserSubstrateRunner(d1OnnxSubstrate)
    loadAbort.current = controller
    pendingLoad.current = loaded
    try {
      await loaded.load(phase => {
        if (requestId === generation.current) {
          setStatus(phase === 'tokenizer'
            ? 'Loading tokenizer…'
            : phase === 'graph'
              ? 'Fetching and compiling ~406 MB Q8 ONNX weights…'
              : 'Ready (WebGPU provider requested)')
        }
      }, controller.signal)
      if (requestId !== generation.current || controller.signal.aborted) {
        await loaded.dispose()
        return
      }
      // Prevent a last-moment Stop click from disposing a successfully adopted session.
      if (pendingLoad.current === loaded) pendingLoad.current = null
      pipeline.current = loaded
    } catch (reason) {
      if (requestId === generation.current) {
        if (controller.signal.aborted) {
          setStatus('Load stopped; model not loaded')
          if ((reason as Error)?.name !== 'AbortError') {
            // Preserve a concurrent network/loader/cleanup failure as evidence.
            setError('Loading also failed: ' + String(reason))
          }
        } else {
          setError(String(reason))
          setStatus('Load failed')
        }
      }
    } finally {
      if (pendingLoad.current === loaded) pendingLoad.current = null
      if (loadAbort.current === controller) loadAbort.current = null
      if (requestId === generation.current) {
        setLoading(false)
        setStopping(false)
      }
    }
  }

  async function stopLoading() {
    const instance = pendingLoad.current
    if (!instance || stopping) return
    const requestId = generation.current
    setStopping(true)
    setStatus('Stopping; waiting for the current ONNX load step and cleanup…')
    loadAbort.current?.abort()
    try {
      // ORT's InferenceSession.create may not interrupt its active download;
      // releasing the late result is the only guarantee here.
      await instance.dispose()
      // The load promise owns the terminal status, including loader/cleanup errors.
    } catch (reason) {
      if (requestId === generation.current) {
        setError('Runtime cleanup failed: ' + String(reason))
        setStatus('Load cleanup failed')
      }
    }
  }

  async function unload() {
    if (loading || deciding) return
    ++generation.current
    const instance = pipeline.current
    pipeline.current = null
    try {
      if (instance) await instance.dispose()
      setStatus('Not loaded')
      setAnswer(null)
      setError('')
    } catch (reason) {
      setError('Runtime disposal failed: ' + String(reason))
    }
  }

  async function decide() {
    const instance = pipeline.current
    if (!instance || loading || deciding) return
    let payload: D1Question
    try {
      payload = kind === 'choice'
        ? { type: 'choice', instructions: question, criteria: parseChoiceOptions(options) }
        : { type: 'noul', instructions: question }
    } catch (reason) {
      setError(String(reason))
      return
    }
    setDeciding(true)
    setError('')
    setAnswer(null)
    try {
      setAnswer(await instance.run({ state, question: payload }))
    } catch (reason) {
      setError(String(reason))
    } finally {
      setDeciding(false)
    }
  }

  return (
    <section className="space-y-4 rounded-2xl border border-outline bg-surface-container p-4 text-on-surface">
      <div className="space-y-1">
        <p className="text-sm text-on-surface-variant">D1 Omni 600M · Q8 ONNX · WebGPU</p>
        <p className="text-sm text-on-surface-variant">
          Real typed text decisions. No image or audio support in this first slice.
          Does not use the LiteRT runtime and does not download until you choose Load.
        </p>
        <p className="text-xs text-on-surface-variant">
          Weights © Liquid AI, LFM Open License v1.0. Pinned ONNX revision:
          {' '}{D1_MODEL_REVISION.slice(0, 12)}. Decision scores are uncalibrated.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={loading || !!pipeline.current}
          onClick={() => { void load() }}
          className="rounded-lg bg-primary px-4 py-2 text-on-primary disabled:opacity-50"
        >Load WebGPU model</button>
        {loading && (
          <button
            type="button"
            disabled={stopping}
            onClick={() => { void stopLoading() }}
            className="rounded-lg border border-outline px-3 py-2 text-sm disabled:opacity-50"
            title="Stop as soon as the current ONNX Runtime loading step permits cleanup"
          >{stopping ? 'Stopping…' : 'Stop loading'}</button>
        )}
        <button
          type="button"
          disabled={loading || deciding || !pipeline.current}
          onClick={() => { void unload() }}
          className="rounded-lg border border-outline px-3 py-2 disabled:opacity-50"
        >Unload</button>
        <span role="status" className="text-sm text-on-surface-variant">{status}</span>
      </div>
      <label className="block space-y-1 text-sm">
        <span>Observed text state</span>
        <textarea
          className="w-full rounded-lg border border-outline bg-surface p-3 text-on-surface"
          rows={3} value={state} onChange={event => setState(event.target.value)}
        />
      </label>
      <label className="block space-y-1 text-sm">
        <span>Question type</span>
        <select
          className="w-full rounded-lg border border-outline bg-surface p-2 text-on-surface"
          value={kind} onChange={event => setKind(event.target.value as 'choice' | 'noul')}
        >
          <option value="choice">Choose among named options</option>
          <option value="noul">Yes / no statement</option>
        </select>
      </label>
      <label className="block space-y-1 text-sm">
        <span>{kind === 'choice' ? 'Question' : 'Statement to evaluate'}</span>
        <input
          className="w-full rounded-lg border border-outline bg-surface p-2 text-on-surface"
          value={question} onChange={event => setQuestion(event.target.value)}
        />
      </label>
      {kind === 'choice' && (
        <label className="block space-y-1 text-sm">
          <span>Choices (one name: description per line; 2–16 choices)</span>
          <textarea
            className="w-full rounded-lg border border-outline bg-surface p-3 text-on-surface"
            rows={4} value={options} onChange={event => setOptions(event.target.value)}
          />
        </label>
      )}
      <button
        type="button" onClick={() => { void decide() }}
        disabled={!pipeline.current || loading || deciding}
        className="rounded-lg bg-primary px-4 py-2 text-on-primary disabled:opacity-50"
      >{deciding ? 'Deciding…' : 'Run real inference'}</button>
      {error && <p role="alert" className="rounded-lg border border-error p-3 text-sm text-error">{error}</p>}
      {answer && (
        <div aria-label="D1 ONNX inference result" className="space-y-2 rounded-xl border border-outline p-3">
          <p className="font-semibold">Decision: {answer.label}</p>
          {Object.entries(answer.probabilities).map(([label, probability]) => (
            <div key={label} className="flex justify-between gap-3 text-sm">
              <span>{label}</span>
              <span className="tabular-nums">{(probability * 100).toFixed(2)}%</span>
            </div>
          ))}
          <p className="text-xs text-on-surface-variant">
            {answer.elapsedMs.toFixed(1)} ms · ONNX Runtime Web · WebGPU requested
            · Raw, uncalibrated probabilities; no action authorization or persistent qualification.
          </p>
        </div>
      )}
      <p className="text-xs text-on-surface-variant">
        Requires browser WebGPU and substantial free GPU memory. No silent WASM fallback.
        Stop loading prevents a late model from becoming ready and releases its resources, but an ONNX download or compilation already in progress may continue until the library call returns. A successful run here is session evidence, not a verified model claim.
      </p>
    </section>
  )
}
