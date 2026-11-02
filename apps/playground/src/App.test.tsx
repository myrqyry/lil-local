import { describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ModelAdapter } from './adapters/types'
import App from './App'

vi.mock('./components/ModelRunner', () => ({
  default: ({ adapters }: { adapters: ModelAdapter[] }) => {
    const bonsai = adapters.find(a => a.modelId === 'bonsai2-2b-onnx-webgpu-candidate')
    const pix2pix = adapters.find(a => a.modelId === 'pix2pix-edges2shoes-webgpu')
    return <section
      data-model-id={bonsai?.modelId}
      data-disabled={String(bonsai?.disabled)}
      data-pix2pix={pix2pix?.modelId}
      data-pix2pix-pipeline={String(pix2pix?.isPipeline)}
    >
      {bonsai?.metadata.name}
      {pix2pix?.metadata.name}
    </section>
  },
}))

describe('catalog special workspaces', () => {
  it('retains Bonsai in the catalog while marking it unavailable for execution', () => {
    const html = renderToStaticMarkup(<App />)
    expect(html).toContain('Bonsai 2')
    expect(html).toContain('data-disabled="true"')
  })

  it('exposes Pix2Pix as an explicit pipeline workspace rather than a LiteRT model', () => {
    const html = renderToStaticMarkup(<App />)
    expect(html).toContain('Pix2Pix')
    expect(html).toContain('data-pix2pix="pix2pix-edges2shoes-webgpu"')
    expect(html).toContain('data-pix2pix-pipeline="true"')
  })
})
