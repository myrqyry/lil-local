import { describe, expect, it, vi } from 'vitest'
import { connectBonsaiNative, completeBonsaiNative, parseBonsaiNativePort } from './bonsai2Native'

describe('Bonsai 2 native runtime (PrismML fork) client', () => {
  it('accepts only validated loopback ports, never arbitrary remote endpoints', () => {
    expect(parseBonsaiNativePort('18086')).toBe(18086)
    expect(() => parseBonsaiNativePort('80')).toThrow(/port/)
    expect(() => parseBonsaiNativePort('0')).toThrow(/port/)
    expect(() => parseBonsaiNativePort('18086/path')).toThrow(/port/)
  })
  it('does not send prompts to a server advertising another model', async () => {
    const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => ({ ok:true, json: async () => ({data:[{id:'qwen3.5:2b'}]}) }))
    await expect(connectBonsaiNative(18086,fetcher as never)).rejects.toThrow(/Bonsai 2 2B/)
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(fetcher.mock.calls[0][0]).toBe('http://127.0.0.1:18086/v1/models')
  })
  it('requires a verified Bonsai 2 2B identity before requesting genuine text output', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce({ok:true,json:async()=>({data:[{id:'Ternary-Bonsai-2-2B-PTQ1_0.gguf'}]})})
      .mockResolvedValueOnce({ok:true,json:async()=>({model:'Ternary-Bonsai-2-2B-PTQ1_0.gguf',
        choices:[{message:{role:'assistant',content:'Actual server response'}}]})})
    const connection = await connectBonsaiNative(18086,fetcher)
    expect(connection.model).toMatch(/2B/)
    const output = await completeBonsaiNative(connection,'Hello',fetcher)
    expect(output).toBe('Actual server response')
    expect(fetcher.mock.calls[1][0]).toBe('http://127.0.0.1:18086/v1/chat/completions')
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toMatchObject({
      model:'Ternary-Bonsai-2-2B-PTQ1_0.gguf',stream:false,max_tokens:96,
      messages:[{role:'user',content:'Hello'}]})
  })
  it('rejects empty prompts, missing server output, or mismatched completion model', async () => {
    const conn={baseUrl:'http://127.0.0.1:18086',model:'Ternary-Bonsai-2-2B-PTQ1_0.gguf'}
    const fetcher=vi.fn(async()=>({ok:true,json:async()=>({model:'qwen3.5:2b',choices:[{message:{content:'wrong'}}]})}))
    await expect(completeBonsaiNative(conn,'',fetcher as never)).rejects.toThrow(/prompt/)
    expect(fetcher).not.toHaveBeenCalled()
    await expect(completeBonsaiNative(conn,'ok',fetcher as never)).rejects.toThrow(/model/)
  })
})
