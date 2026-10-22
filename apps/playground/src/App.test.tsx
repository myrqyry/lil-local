import { describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ModelAdapter } from './adapters/types'
import App from './App'

vi.mock('./components/ModelRunner', () => ({
  default: ({ adapters }: { adapters: ModelAdapter[] }) => {
    const bonsai = adapters.find(a => a.modelId === 'bonsai2-2b-onnx-webgpu-candidate')
    return <section data-model-id={bonsai?.modelId} data-disabled={String(bonsai?.disabled)}>
      {bonsai?.metadata.name}
    </section>
  },
}))

describe('Bonsai candidate availability', () => {
  it('retains Bonsai in the catalog while marking it unavailable for execution', () => {
    const html = renderToStaticMarkup(<App />)
    expect(html).toContain('Bonsai 2')
    expect(html).toContain('data-disabled="true"')
  })
})
