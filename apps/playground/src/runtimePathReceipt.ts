import type { RuntimePathProof } from './runtimePathProof'

export interface RuntimePathReceipt {
  schemaVersion: 1
  evidenceKind: 'current-session-runtime-path'
  proof: RuntimePathProof
  interpretation: string
}

/** Serialize observed session telemetry without implying durable qualification. */
export function serializeRuntimePathReceipt(proof: RuntimePathProof): string {
  const receipt: RuntimePathReceipt = {
    schemaVersion: 1,
    evidenceKind: 'current-session-runtime-path',
    proof,
    interpretation: 'This receipt records one browser session. It does not promote durable model verification.',
  }
  return `${JSON.stringify(receipt, null, 2)}\n`
}

/** Download only the user-requested session evidence, not a qualification record. */
export function downloadRuntimePathReceipt(proof: RuntimePathProof): void {
  const blob = new Blob([serializeRuntimePathReceipt(proof)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  const safeModelId = proof.modelId
    .replace(/[^a-zA-Z0-9._-]+/g, '-')
    .replace(/^\.+/, '')
    .slice(0, 96) || 'model'

  anchor.href = url
  anchor.download = `${safeModelId}-runtime-session.json`
  try {
    document.body.appendChild(anchor)
    anchor.click()
  } finally {
    anchor.remove()
    // A zero-delay revoke can invalidate downloads before slower browsers begin them.
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
  }
}
