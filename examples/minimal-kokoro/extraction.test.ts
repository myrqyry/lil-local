import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('minimal Kokoro extraction', () => {
  it('uses only public package imports and exposes browser audio output', () => {
    const source = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
    expect(source).toContain('@lil-local/kokoro')
    expect(source).toContain('AudioContext')
    expect(source).toContain('receipt')
    expect(source).not.toMatch(/\.\.\/\.\/src|apps\/playground|podqast/)
  })
})
