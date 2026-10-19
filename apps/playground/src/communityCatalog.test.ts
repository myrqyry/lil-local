import { describe, expect, it } from 'vitest'
import type { ModelAdapter } from './adapters/types'
import {
  communityCatalogUrl,
  findCommunityAdapter,
  parseCommunityModels,
} from './communityCatalog'

function adapter(id: string, modelPath: string): ModelAdapter {
  return {
    modelId: id,
    metadata: { name: id, description: '', modelPath, tags: [] },
    inputSpecs: [],
    outputSpecs: [],
    prepareInputs: () => ({}),
    parseOutputs: async () => ({}),
  }
}

describe('LiteRT Community discovery boundary', () => {
  it('constructs bounded, explicitly filtered Hugging Face searches', () => {
    const url = new URL(communityCatalogUrl('moonshine + tiny', 50))
    expect(url.origin).toBe('https://huggingface.co')
    expect(url.pathname).toBe('/api/models')
    expect(url.searchParams.get('author')).toBe('litert-community')
    expect(url.searchParams.get('search')).toBe('moonshine + tiny')
    expect(url.searchParams.get('sort')).toBe('downloads')
    expect(url.searchParams.get('limit')).toBe('40')
    expect(url.searchParams.get('skip')).toBe('50')
    expect(() => communityCatalogUrl('', -1)).toThrow()
  })

  it('only accepts canonical public organization IDs, sanitizes optional metadata and de-duplicates', () => {
    const result = parseCommunityModels([
      { id: 'litert-community/moonshine-tiny', downloads: 100, pipeline_tag: 'automatic-speech-recognition' },
      { id: 'litert-community/moonshine-tiny', downloads: 1000 },
      { id: 'litert-community/secret', private: true },
      { id: 'other-org/gemma-4', downloads: 9999 },
      { id: 'litert-community/foo/../../bad' },
      { id: 'litert-community/..' },
      { id: 'litert-community/valid-B', downloads: -5, gated: 'manual', pipeline_tag: 42 },
      { id: 'litert-community/missing', downloads: Infinity },
      null,
    ])
    expect(result).toEqual([
      {
        id: 'litert-community/moonshine-tiny',
        name: 'moonshine-tiny',
        url: 'https://huggingface.co/litert-community/moonshine-tiny',
        downloads: 100,
        pipelineTag: 'automatic-speech-recognition',
        gated: false,
      },
      {
        id: 'litert-community/valid-B',
        name: 'valid-B',
        url: 'https://huggingface.co/litert-community/valid-B',
        downloads: null,
        pipelineTag: null,
        gated: true,
      },
      {
        id: 'litert-community/missing',
        name: 'missing',
        url: 'https://huggingface.co/litert-community/missing',
        downloads: null,
        pipelineTag: null,
        gated: false,
      },
    ])
    expect(parseCommunityModels({ error: 'rate limited' })).toEqual([])
  })

  it('matches only existing HF-based runnable adapter entries, not similarly named paths or pipeline-only entries', () => {
    const candidates = [
      adapter('wrong-host', 'https://huggingface.co.evil.test/litert-community/moonshine-tiny/resolve/main/model.tflite'),
      adapter('pipeline', ''),
      adapter('other-repo', 'https://huggingface.co/other-org/moonshine-tiny/resolve/main/model.tflite'),
      adapter('moonshine-asr', 'https://huggingface.co/litert-community/moonshine-tiny/resolve/main/moonshine_tiny.tflite'),
    ]
    expect(findCommunityAdapter('litert-community/moonshine-tiny', candidates)?.modelId).toBe('moonshine-asr')
    expect(findCommunityAdapter('litert-community/absent', candidates)).toBeNull()
    expect(findCommunityAdapter('litert-community/moonshine-tiny-extra', candidates)).toBeNull()
  })
})
