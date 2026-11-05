import type { BrowserSubstrate } from '@lil-local/inference-core'
import type { LanguageDetector } from '@mediapipe/tasks-text'

/**
 * The publisher's immutable versioned model; do not replace with 'latest'.
 * Verified against bytes downloaded from Google's storage on 2026-11-04.
 */
export const LANGUAGE_DETECTOR_MODEL = {
  url: 'https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite',
  bytes: 315_294,
  sha256: '7db4f23dfe1ad8966b050b419a865da451143fd43eb6b606a256aadeeb1e5417',
} as const

/** SDK-compatible versioned WASM. Override to self-host exactly this version. */
export const MEDIAPIPE_TEXT_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@1.2.2/wasm'

export type TextLoadProgress = 'model' | 'integrity' | 'wasm'

export interface LanguageDetectorOptions {
  /** Only the official SHA-verified weights may be used. */
  model?: Blob | Uint8Array
  /** Same-version self-hosted WASM directory. */
  wasmBase?: string
  /** GPU is MediaPipe WebGL, never native WebGPU. */
  delegate?: 'GPU' | 'CPU'
  maxResults?: number
  scoreThreshold?: number
}
export interface LanguageDetectionResult {
  kind: 'language-detection'
  languages: {languageCode:string;probability:number}[]
}

export function validateLanguageText(text:string):string {
  if(typeof text!=='string')throw new Error('Language detector requires a string')
  if(!text.trim())throw new Error('Language detector text cannot be empty')
  // Bounded untrusted user input; code units, not bytes or tokens.
  if(text.length>32_768)throw new Error('Language detector text length exceeds 32768 characters')
  return text
}

/** Do not guess languages, translate, normalize or reorder the native results. */
export function mapLanguageDetectorResult(
  native:{languages:readonly {languageCode:string;probability:number}[]},
):LanguageDetectionResult {
  if(!native||!Array.isArray(native.languages))
    throw new Error('Language detector result must include languages array')
  return {kind:'language-detection',languages:native.languages.map(value=>{
    if(typeof value?.languageCode!=='string'||!value.languageCode.trim())
      throw new Error('Language detector returned invalid language code')
    if(!Number.isFinite(value.probability)||value.probability<0||value.probability>1)
      throw new Error('Language detector probability must be finite in [0,1]')
    return {languageCode:value.languageCode,probability:value.probability}
  })}
}

export async function verifyLanguageModelBytes(
  bytes:Uint8Array,
  expected:{bytes:number;sha256:string}=LANGUAGE_DETECTOR_MODEL,
):Promise<void> {
  if(bytes.byteLength!==expected.bytes)throw new Error('MediaPipe text model size mismatch')
  if(!globalThis.crypto?.subtle)
    throw new Error('MediaPipe text model SHA-256 requires secure WebCrypto context')
  const digest=await crypto.subtle.digest('SHA-256',Uint8Array.from(bytes))
  const got=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('')
  if(got!==expected.sha256)throw new Error('MediaPipe text model SHA-256 mismatch')
}

function abortError():Error {
  const err=new Error('MediaPipe text loading aborted')
  err.name='AbortError'
  return err
}
function checkAbort(signal?:AbortSignal):void{
  if(signal?.aborted)throw abortError()
}

async function languageModelBytes(
  options:LanguageDetectorOptions,
  report?:(event:TextLoadProgress)=>void,
  signal?:AbortSignal,
):Promise<Uint8Array> {
  checkAbort(signal)
  report?.('model')
  let bytes:Uint8Array
  if(options.model instanceof Uint8Array)bytes=options.model
  else if(options.model instanceof Blob)bytes=new Uint8Array(await options.model.arrayBuffer())
  else {
    const response=await fetch(LANGUAGE_DETECTOR_MODEL.url,{signal})
    if(!response.ok)throw new Error('MediaPipe language model download HTTP '+response.status)
    const declared=response.headers.get('content-length')
    if(declared && Number(declared)!==LANGUAGE_DETECTOR_MODEL.bytes)
      throw new Error('MediaPipe language model size mismatch in response')
    bytes=new Uint8Array(await response.arrayBuffer())
  }
  checkAbort(signal)
  report?.('integrity')
  await verifyLanguageModelBytes(bytes)
  checkAbort(signal)
  return bytes
}

/**
 * Dedicated text-task substrate. No automatic language routing or translation:
 * scores are native per-whole-input probabilities, not span segmentation.
 *
 * Default GPU means MediaPipe's WebGL delegate; no WebGPU path is claimed.
 * If initialization fails, the error is surfaced, not silently masked with CPU.
 */
export function createLanguageDetectorSubstrate(
  options:LanguageDetectorOptions={},
):BrowserSubstrate<string,LanguageDetectionResult,TextLoadProgress>{
  const maxResults=options.maxResults??3
  if(!Number.isSafeInteger(maxResults)||maxResults<1||maxResults>20)
    throw new Error('maxResults must be an integer in [1,20]')
  const scoreThreshold=options.scoreThreshold??0
  if(!Number.isFinite(scoreThreshold)||scoreThreshold<0||scoreThreshold>1)
    throw new Error('scoreThreshold must be a finite probability in [0,1]')
  const delegate=options.delegate??'GPU'
  if(delegate!=='GPU'&&delegate!=='CPU')
    throw new Error('MediaPipe text delegate must be GPU (WebGL) or CPU')
  return {
    id:'mediapipe:language-detector-float32-1',
    format:'tflite',
    capabilities:['text-classification'],
    // MediaPipe uses WASM infrastructure even when the internal delegate
    // is WebGL; it is NOT an @lil-local/inference-core WebGPU backend.
    requestedBackends:['wasm'],
    async create(progress,signal){
      const bytes=await languageModelBytes(options,progress,signal)
      progress?.('wasm')
      const {FilesetResolver,LanguageDetector}=await import('@mediapipe/tasks-text')
      checkAbort(signal)
      const fileset=await FilesetResolver.forTextTasks(
        options.wasmBase??MEDIAPIPE_TEXT_WASM_BASE,
      )
      checkAbort(signal)
      let task:LanguageDetector|null=null
      try{
        task=await LanguageDetector.createFromOptions(fileset,{
          baseOptions:{modelAssetBuffer:bytes,delegate},
          maxResults,scoreThreshold,
        })
        checkAbort(signal)
        const native=task
        task=null
        return {
          async infer(text){
            const validated=validateLanguageText(text)
            return mapLanguageDetectorResult(native.detect(validated))
          },
          async dispose(){native.close()},
        }
      } finally {task?.close()}
    },
  }
}
