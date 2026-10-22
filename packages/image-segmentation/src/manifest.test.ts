import { describe, expect, it } from 'vitest'
import { u2NetSaliencyManifest } from './pipeline'

describe('source-pinned U²-Net saliency candidate', () => {
  it('pins the exact 88MB model and never elevates artifact existence to an inference pass', () => {
    expect(u2NetSaliencyManifest.assets[0]).toMatchObject({
      bytes: 88_230_272,
      sha256: 'dd338f190a538ca3de9792b20b7617038cd56f8e440f38ff25da5554f32b9df2',
    })
    expect(u2NetSaliencyManifest.assets[0].path).toContain('/resolve/defc203955a46f3cc760deb8c76d96f10331a46c/')
    expect(u2NetSaliencyManifest.verification).toMatchObject({
      assets:'untested',compile:'untested',inference:'untested',output:'untested',qualification:'limited',
    })
    expect(u2NetSaliencyManifest.backends).toEqual({wasm:'experimental'})
  })
})
