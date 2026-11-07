import { describe, expect, it } from 'vitest'
import { createAudio8Manifest, AUDIO8_SOURCE_REVISION, audio8Profiles } from './manifest'

describe('Audio8 LiteRT candidate assets', () => {
  it('pins the publisher revision and real integrity metadata for each graph', () => {
    const manifest = createAudio8Manifest()
    expect(AUDIO8_SOURCE_REVISION).toBe('752cd650bf5f649f2c0139ea02ac5627e9e60a5a')
    expect(manifest.verification).toMatchObject({
      assets: 'untested', compile: 'untested', inference: 'untested',
      output: 'untested', qualification: 'unverified',
      upstreamRevision: AUDIO8_SOURCE_REVISION,
    })
    expect(manifest.assets.find(asset => asset.id === 'slow-ar')).toMatchObject({
      path: 'slow_ar_int8.tflite',
      bytes: 558_303_696,
      sha256: '8a54c2a1d331e0bb9efa42c98506672ab7739a1d9a77cf54f7d383c8903d882e',
    })
    expect(manifest.assets.every(asset => !!asset.sha256 && /^[a-f0-9]{64}$/.test(asset.sha256))).toBe(true)
  })

  it('provides distinct synthesis and voice registration downloads', () => {
    const synthesis = createAudio8Manifest(audio8Profiles.synthesis)
    const cloning = createAudio8Manifest(audio8Profiles.voiceRegistration)
    expect(synthesis.assets.some(asset => asset.id === 'voice-encoder')).toBe(false)
    expect(cloning.assets.some(asset => asset.id === 'voice-encoder')).toBe(true)
    expect(synthesis.assets.some(asset => asset.id === 'codec-192')).toBe(false)
    expect(cloning.assets.some(asset => asset.id === 'codec-192')).toBe(false)
    expect(cloning.memory.downloadBytes - synthesis.memory.downloadBytes).toBe(419_269_280)
    for (const manifest of [synthesis, cloning]) {
      expect(manifest.memory.downloadBytes).toBe(manifest.assets.reduce((total, asset) => total + (asset.bytes ?? 0), 0))
      expect(manifest.backends).toEqual({ wasm: 'experimental', webgpu: 'experimental' })
    }
  })

  it('keeps the larger decoder an explicit opt-in and includes bundled voice assets', () => {
    const extended = createAudio8Manifest(audio8Profiles.extendedDecode)
    expect(extended.assets.some(asset => asset.id === 'codec-192')).toBe(true)
    expect(extended.assets.some(asset => asset.id === 'voice-encoder')).toBe(false)
    expect(extended.assets.filter(asset => asset.id.startsWith('voice-'))).toHaveLength(4)
  })
})
