import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { Bonsai2CandidatePanel } from './Bonsai2CandidatePanel'

describe('Bonsai 2 honest capability panel', () => {
  it('blocks known Scan decoder, offers only explicit native connections and no phantom generated content', () => {
    const html = renderToStaticMarkup(<Bonsai2CandidatePanel />)
    expect(html).toContain('18 ONNX Scan operations')
    expect(html).toContain('Connect to local Bonsai server')
    expect(html).toContain('127.0.0.1')
    expect(html).not.toContain('Generate with connected Bonsai 2B')
    expect(html).toContain('no model load')
  })
})
