import {describe,expect,it,vi,beforeEach} from 'vitest'
import {BrowserSubstrateRunner} from '@lil-local/inference-core'
import {
  LANGUAGE_DETECTOR_MODEL,MEDIAPIPE_TEXT_WASM_BASE,
  createLanguageDetectorSubstrate,verifyLanguageModelBytes,
  validateLanguageText,mapLanguageDetectorResult,
} from './index'

const sdk=vi.hoisted(()=>({
  create:vi.fn(),detect:vi.fn(),close:vi.fn(),fileset:vi.fn(),
}))
vi.mock('@mediapipe/tasks-text',()=>({
  FilesetResolver:{forTextTasks:sdk.fileset},
  LanguageDetector:{createFromOptions:sdk.create},
}))
beforeEach(()=>{
  vi.clearAllMocks()
  sdk.fileset.mockResolvedValue({})
  sdk.create.mockResolvedValue({detect:sdk.detect,close:sdk.close})
  sdk.detect.mockReturnValue({languages:[{languageCode:'es',probability:.97}]})
})

describe('MediaPipe text language detector',()=>{
  it('uses publisher pinned model and same-version WASM, not a mutable latest',()=>{
    expect(LANGUAGE_DETECTOR_MODEL.bytes).toBe(315_294)
    expect(LANGUAGE_DETECTOR_MODEL.sha256)
      .toBe('7db4f23dfe1ad8966b050b419a865da451143fd43eb6b606a256aadeeb1e5417')
    expect(LANGUAGE_DETECTOR_MODEL.url)
      .toContain('language_detector/language_detector/float32/1/language_detector.tflite')
    expect(MEDIAPIPE_TEXT_WASM_BASE).toContain('@mediapipe/tasks-text@1.2.2/wasm')
  })
  it('validates real SHA-256 content and exact byte length, including changed bytes',async()=>{
    const bytes=Uint8Array.of(10,20,30)
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
      n=>n.toString(16).padStart(2,'0')).join('')
    await expect(verifyLanguageModelBytes(bytes,{bytes:3,sha256:hash})).resolves.toBeUndefined()
    await expect(verifyLanguageModelBytes(Uint8Array.of(10,20,31),{bytes:3,sha256:hash}))
      .rejects.toThrow(/SHA-256/)
    await expect(verifyLanguageModelBytes(bytes,{bytes:4,sha256:hash}))
      .rejects.toThrow(/size/)
  })
  it('preserves language codes, native probabilities, and empty/uncertain outputs',()=>{
    expect(mapLanguageDetectorResult({languages:[
      {languageCode:'es',probability:.93},{languageCode:'en',probability:.06},
    ]})).toEqual({kind:'language-detection',languages:[
      {languageCode:'es',probability:.93},{languageCode:'en',probability:.06},
    ]})
    expect(mapLanguageDetectorResult({languages:[]})).toEqual({kind:'language-detection',languages:[]})
    expect(()=>mapLanguageDetectorResult({languages:[{languageCode:'en',probability:NaN}]}))
      .toThrow(/probability/)
    expect(()=>mapLanguageDetectorResult({languages:[{languageCode:'',probability:.4}]}))
      .toThrow(/code/)
  })
  it('rejects empty, excessive or non-text inputs before task inference',()=>{
    expect(validateLanguageText('  ¡Hola!  ')).toBe('  ¡Hola!  ')
    expect(()=>validateLanguageText('   ')).toThrow(/empty/)
    expect(()=>validateLanguageText('x'.repeat(32769))).toThrow(/length/)
    expect(()=>validateLanguageText({} as never)).toThrow(/string/)
  })
  it('validates options and only allows real SDK delegates',()=>{
    expect(()=>createLanguageDetectorSubstrate({maxResults:0})).toThrow(/maxResults/)
    expect(()=>createLanguageDetectorSubstrate({maxResults:2.5})).toThrow(/maxResults/)
    expect(()=>createLanguageDetectorSubstrate({scoreThreshold:NaN})).toThrow(/scoreThreshold/)
    expect(()=>createLanguageDetectorSubstrate({scoreThreshold:1.1})).toThrow(/scoreThreshold/)
    expect(()=>createLanguageDetectorSubstrate({delegate:'WebGPU' as never})).toThrow(/delegate/)
    expect(()=>createLanguageDetectorSubstrate({delegate:'GPU'})).not.toThrow()
    expect(()=>createLanguageDetectorSubstrate({delegate:'CPU'})).not.toThrow()
  })
  it('constructs cold and a pre-aborted load never fetches',async()=>{
    const fetcher=vi.spyOn(globalThis,'fetch')
    try{
      const runner=new BrowserSubstrateRunner(createLanguageDetectorSubstrate())
      expect(runner.status).toBe('idle')
      expect(runner.substrate.capabilities).toContain('text-classification')
      expect(fetcher).not.toHaveBeenCalled()
      const abort=new AbortController();abort.abort()
      await expect(runner.load(undefined,abort.signal)).rejects.toMatchObject({name:'AbortError'})
      expect(fetcher).not.toHaveBeenCalled()
      expect(runner.status).toBe('idle')
    }finally{fetcher.mockRestore()}
  })
  it('loads once and detects repeated strings via one real runner API boundary, then closes',async()=>{
    const fake=new Uint8Array(LANGUAGE_DETECTOR_MODEL.bytes)
    const realDigest=crypto.subtle.digest.bind(crypto.subtle)
    const sha=LANGUAGE_DETECTOR_MODEL.sha256
    const digest=vi.spyOn(crypto.subtle,'digest').mockResolvedValue(
      Uint8Array.from(sha.match(/../g)!.map(x=>parseInt(x,16))).buffer as ArrayBuffer)
    const runner=new BrowserSubstrateRunner(createLanguageDetectorSubstrate({
      model:fake,delegate:'CPU',scoreThreshold:0,maxResults:3,
    }))
    try{
      await runner.load()
      expect(sdk.create).toHaveBeenCalledTimes(1)
      expect(sdk.create.mock.calls[0][1]).toMatchObject({
        baseOptions:{delegate:'CPU'},maxResults:3,scoreThreshold:0,
      })
      expect(await runner.run('Esta es una frase larga en español.')).toEqual({
        kind:'language-detection',languages:[{languageCode:'es',probability:.97}],
      })
      await runner.run('A long English sentence for testing.')
      expect(sdk.detect).toHaveBeenCalledTimes(2)
      await expect(runner.run(' ')).rejects.toThrow(/empty/)
    }finally{
      await runner.dispose()
      digest.mockRestore()
    }
    expect(sdk.close).toHaveBeenCalledTimes(1)
    expect(realDigest).toBeTypeOf('function')
  })
})
