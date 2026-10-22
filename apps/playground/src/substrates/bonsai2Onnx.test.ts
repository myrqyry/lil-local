import { describe, expect, it } from 'vitest'
import {
  assessBonsaiGpu,
  BONSAI2_REPO,
  BONSAI2_REVISION,
  BONSAI2_DECODER_AUDIT,
  assessBonsaiBrowserPath,
} from './bonsai2Onnx'

describe('Ternary Bonsai 2 WebGPU hardware qualification', () => {
  it('rejects an adapter without the required shader-f16 feature', () => {
    const result = assessBonsaiGpu(new Set(['timestamp-query']), {
      vendor: 'nvidia', architecture: 'turing',
    })
    expect(result).toMatchObject({
      state: 'unsupported', vendor: 'nvidia', architecture: 'turing',
      shaderF16: false,
    })
    expect(result.reason).toMatch(/GatherBlockQuantized/)
    expect(result.reason).toMatch(/shader-f16/)
  })

  it('reports only eligibility—not inferred inference success—when f16 exists', () => {
    const result = assessBonsaiGpu(new Set(['shader-f16']), {
      vendor: 'other', architecture: 'newer-device',
    })
    expect(result.state).toBe('eligible-for-qualification')
    expect(result.shaderF16).toBe(true)
    expect(result.reason).toMatch(/unverified/)
  })

  it('fails closed without known features or adapter identity', () => {
    const result = assessBonsaiGpu(new Set(), {})
    expect(result.state).toBe('unsupported')
    expect(result.vendor).toBe('unknown')
    expect(result.architecture).toBe('unknown')
  })

  it('pins the observed source repo and revision, not a moving main branch', () => {
    expect(BONSAI2_REPO).toBe('danyalmalik/Ternary-Bonsai-2-2B-ONNX-WebGPU')
    expect(BONSAI2_REVISION).toMatch(/^[0-9a-f]{40}$/)
  })
})

describe('pinned decoder structural qualification', () => {
  it('refuses the real published 2B decoder even when shader-f16 is available', () => {
    expect(BONSAI2_DECODER_AUDIT).toMatchObject({
      sha256:'a4b3e8a9633cfffa35104ccbf2bd748c26f36cd8b1281c8443d47b82ecd0268f',
      scanNodes:18, loopNodes:0, status:'failed-webgpu-audit',
    })
    const path = assessBonsaiBrowserPath(new Set(['shader-f16']),{vendor:'nvidia',architecture:'turing'})
    expect(path.state).toBe('blocked-unsafe-decoder')
    expect(path.reason).toMatch(/Scan/)
    expect(path.hardware.shaderF16).toBe(true)
  })
  it('never treats missing fp16 support as model readiness', () => {
    const path=assessBonsaiBrowserPath(new Set(),{vendor:'nvidia'})
    expect(path.state).toBe('blocked-unsafe-decoder')
    expect(path.hardware.state).toBe('unsupported')
  })
})
