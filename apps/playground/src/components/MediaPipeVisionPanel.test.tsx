import {describe,expect,it} from 'vitest'
import {renderToStaticMarkup} from 'react-dom/server'
import {MediaPipeVisionPanel} from './MediaPipeVisionPanel'
describe('existing live vision workbench exposes conditioning models',()=>{
 it('allows opt-in selfie foreground masks without replacing the camera workflow',()=>{
  const html=renderToStaticMarkup(<MediaPipeVisionPanel/>)
  expect(html).toContain('value="person"')
  expect(html).toContain('Selfie')
  expect(html).toContain('value="depthV2"')
  expect(html).toContain('Depth Anything V2')
  expect(html).toContain('Start local vision')
  expect(html).toContain('Image for local vision analysis')
 })
})
