import {describe, expect, it} from 'vitest'
import {renderToStaticMarkup} from 'react-dom/server'
import {Gemma4Panel} from './Gemma4Panel'

describe('Gemma 4 browser experience', () => {
  it('offers pinned E2B and E4B text inference without exposing broken native modalities', () => {
    const html = renderToStaticMarkup(<Gemma4Panel />)
    expect(html).toContain('value="e2b"')
    expect(html).toContain('value="e4b"')
    expect(html).toContain('Gemma prompt')
    expect(html).toContain('New chat')
    expect(html).toContain('Enable thinking')
    expect(html).toContain('Chat stays in memory')
    expect(html).toContain('Gemma chat history')
    expect(html).toContain('Load Gemma locally')
    expect(html).toContain('2.01 GB')
    expect(html).toContain('text-only')
    expect(html).not.toContain('Gemma image input')
    expect(html).not.toContain('Gemma WAV audio input')
    expect(html).not.toContain('Gemma input mode')
  })
})
