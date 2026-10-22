import { useState } from 'react'
import {
  BONSAI2_REPO,
  BONSAI2_REVISION,
  BONSAI2_DECODER_AUDIT,
  inspectBonsaiGpu,
  assessBonsaiBrowserPath,
  type BonsaiGpuEligibility,
} from '../substrates/bonsai2Onnx'
import {
  completeBonsaiNative,
  connectBonsaiNative,
  parseBonsaiNativePort,
  type BonsaiNativeConnection,
} from '../substrates/bonsai2Native'

/** Actual capability UI: published ONNX decoder is blocked; verified native endpoint is opt-in. */
export function Bonsai2CandidatePanel() {
  const [hardware, setHardware] = useState<BonsaiGpuEligibility | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [port, setPort] = useState('18086')
  const [connection, setConnection] = useState<BonsaiNativeConnection | null>(null)
  const [prompt, setPrompt] = useState('')
  const [answer, setAnswer] = useState<string | null>(null)
  const [nativeBusy, setNativeBusy] = useState(false)
  const [nativeError, setNativeError] = useState<string | null>(null)
  const browserPath = assessBonsaiBrowserPath(
    hardware?.shaderF16 ? new Set(['shader-f16']) : new Set(),
    { vendor: hardware?.vendor, architecture: hardware?.architecture },
  )

  async function check() {
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      setHardware(await inspectBonsaiGpu())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      setHardware(null)
    } finally {
      setBusy(false)
    }
  }

  async function connectNative() {
    if (nativeBusy) return
    setNativeBusy(true)
    setNativeError(null)
    setAnswer(null)
    setConnection(null)
    try {
      const connected = await connectBonsaiNative(parseBonsaiNativePort(port))
      setConnection(connected)
    } catch (cause) {
      setNativeError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setNativeBusy(false)
    }
  }

  async function sendNativePrompt() {
    if (!connection || nativeBusy) return
    setNativeBusy(true)
    setNativeError(null)
    setAnswer(null)
    try {
      setAnswer(await completeBonsaiNative(connection, prompt))
    } catch (cause) {
      setNativeError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setNativeBusy(false)
    }
  }

  return (
    <section className="space-y-5 rounded-xl border border-outline-variant bg-surface-container p-4">
      <h2 className="text-lg font-semibold text-on-surface">Ternary Bonsai 2 · 2B</h2>
      <p className="text-sm text-on-surface-variant">
        An experimental compact local text model. The original browser ONNX decoder
        failed a structural GPU audit. Native PrismML GGUF inference is a separate,
        optional route that requires a running local server.
      </p>

      <div className="space-y-2 rounded-lg border border-outline-variant p-3 text-sm">
        <h3 className="font-semibold text-on-surface">Browser ONNX · blocked by graph audit</h3>
        <p className="text-on-surface-variant">
          The pinned decoder contains <strong>{BONSAI2_DECODER_AUDIT.scanNodes} ONNX Scan operations</strong>.
          ONNX Runtime WebGPU cannot execute these as GPU kernels; token prefill can
          fall back to expensive CPU/GPU round trips. Having shader-f16 is not
          enough to qualify this path. No model-weight download or generation is offered.
        </p>
        <p className="text-xs text-on-surface-variant">
          Exact graph audit: {BONSAI2_DECODER_AUDIT.size.toLocaleString()} bytes,
          SHA-256 {BONSAI2_DECODER_AUDIT.sha256.slice(0, 16)}…
        </p>
        <p className="text-xs text-on-surface-variant">
          Source: <a className="underline" href={'https://huggingface.co/' + BONSAI2_REPO}
            target="_blank" rel="noopener noreferrer">{BONSAI2_REPO}</a>
          {' · '}revision {BONSAI2_REVISION.slice(0, 12)}
        </p>
        <button type="button"
          className="rounded-lg border border-outline px-3 py-2 text-sm text-on-surface disabled:opacity-50"
          onClick={() => void check()} disabled={busy}>
          {busy ? 'Checking adapter…' : 'Inspect WebGPU hardware (no model load)'}
        </button>
        {hardware && (
          <div role="status" className="space-y-1 text-sm text-on-surface">
            <p>GPU: {hardware.vendor} / {hardware.architecture};
              shader-f16: <strong>{hardware.shaderF16 ? 'available' : 'unavailable'}</strong></p>
            <p>{browserPath.reason}</p>
          </div>
        )}
        {error && <p role="alert" className="text-sm text-error">{error}</p>}
      </div>

      <div className="space-y-3 rounded-lg border border-outline-variant p-3 text-sm">
        <h3 className="font-semibold text-on-surface">Native PrismML · local server</h3>
        <p className="text-on-surface-variant">
          Run the <a className="underline" href="https://github.com/PrismML-Eng/Bonsai-demo"
            target="_blank" rel="noopener noreferrer">official PrismML llama.cpp fork</a> with
          a Bonsai 2 2B GGUF model, bound to loopback with CORS enabled for this page.
          This app does not install, launch, or download it.
          The first isolated CPU acceptance test of the official 2B PTQ1 GGUF
          aborted on a GGML tensor-layout assertion, so this exact native
          combination is NOT yet qualified. The server must advertise the
          actual Bonsai model before text requests are enabled.
        </p>
        <label className="flex flex-col gap-1 text-sm text-on-surface">
          Native server port on 127.0.0.1
          <input value={port}
            onChange={event => { setPort(event.target.value); setConnection(null); setAnswer(null) }}
            disabled={nativeBusy} inputMode="numeric"
            className="w-32 rounded-lg border border-outline bg-surface px-3 py-2 text-on-surface" />
        </label>
        <button type="button"
          className="rounded-lg bg-primary px-4 py-2 font-medium text-on-primary disabled:opacity-50"
          onClick={() => void connectNative()} disabled={nativeBusy}>
          {nativeBusy ? 'Connecting…' : 'Connect to local Bonsai server'}
        </button>
        {connection && <p role="status" className="text-sm text-on-surface">
          Server advertised: <strong>{connection.model}</strong>. This confirms identity,
          not that inference is working.
        </p>}
        {connection && (
          <div className="space-y-2">
            <label className="flex flex-col gap-1 text-sm text-on-surface">
              Prompt (96 new tokens maximum)
              <textarea value={prompt} maxLength={4096}
                onChange={event => setPrompt(event.target.value)}
                rows={3} className="w-full rounded-lg border border-outline bg-surface px-3 py-2 text-on-surface" />
            </label>
            <button type="button"
              className="rounded-lg bg-primary px-4 py-2 font-medium text-on-primary disabled:opacity-50"
              onClick={() => void sendNativePrompt()} disabled={nativeBusy || !prompt.trim()}>
              {nativeBusy ? 'Generating…' : 'Generate with connected Bonsai 2B'}
            </button>
          </div>
        )}
        {answer && <div className="rounded-md bg-surface p-3 text-sm text-on-surface">
          <strong>Native response</strong><p className="whitespace-pre-wrap">{answer}</p>
        </div>}
        {nativeError && <p role="alert" className="text-sm text-error">
          {nativeError}. Ensure the Prism server is running and its CORS settings allow this page.
        </p>}
      </div>
    </section>
  )
}
