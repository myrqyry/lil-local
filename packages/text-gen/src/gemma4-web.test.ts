import {describe, expect, it} from 'vitest'
import {gemma4E2bWebManifest, gemma4E4bWebManifest} from './manifest'
import {getPinnedGemma4WebSource} from './gemma4-web'

describe('Official Gemma 4 browser artifacts', () => {
  it('selects the explicitly pinned E2B WebGPU artifact, not the native mobile model', () => {
    const manifest = gemma4E2bWebManifest
    expect(manifest.assets[0]).toMatchObject({
      bytes: 2_008_432_640,
      sha256: '3a08e8d94e23b814ae5414469c370c503813949acb8ceaa17e4ebf8a35af35b5',
    })
    expect(manifest.backends).toEqual({webgpu: true})
    expect(manifest.verification).toMatchObject({
      assets: 'pass', compile: 'pass', inference: 'pass', output: 'pass', qualification: 'limited',
      upstreamRevision: 'b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1',
      environments: [expect.objectContaining({browser: expect.stringContaining('Chromium'),backend: 'webgpu'})],
    })
    expect(getPinnedGemma4WebSource(manifest)).toBe(
      'hf://litert-community/gemma-4-E2B-it-litert-lm@b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it-web.litertlm'
    )
  })
  it('pins the WebGPU E4B artifact independently of its general native model', () => {
    expect(gemma4E4bWebManifest.assets[0]).toMatchObject({
      bytes: 2_969_059_328,
      sha256: '3904d826d5dddd25ea173e85204caec09e68ba038116e9b992b69cbdc94f57a0',
    })
    expect(gemma4E4bWebManifest.verification).toMatchObject({
      assets: 'pass', compile: 'pass', inference: 'pass', output: 'pass', qualification: 'limited',
      upstreamRevision: '2eee7ac325f20eb8c9ac1d0e972f7c84663062da',
    })
    expect(getPinnedGemma4WebSource(gemma4E4bWebManifest)).toBe(
      'hf://litert-community/gemma-4-E4B-it-litert-lm@2eee7ac325f20eb8c9ac1d0e972f7c84663062da/gemma-4-E4B-it-web.litertlm'
    )
  })
  it('rejects unpinned, non-web or malformed model assets', () => {
    for(const path of [
      'litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm',
      'litert-community/gemma-4-E2B-it-litert-lm/resolve/abc/gemma-4-E2B-it-web.litertlm',
      'litert-community/gemma-4-E2B-it-litert-lm/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it.litertlm',
    ]) {
      expect(() => getPinnedGemma4WebSource({
        ...gemma4E2bWebManifest, assets: [{...gemma4E2bWebManifest.assets[0], path}],
      })).toThrow(/web|pinned|revision/)
    }
  })
})
