import type { ModelAsset, ModelManifest } from '@lil-local/inference-core'

/** Publisher conversion revision, not a browser inference qualification. */
export const AUDIO8_SOURCE_REPOSITORY = 'litert-community/Audio8-TTS-Realistic-0.6b'
export const AUDIO8_SOURCE_REVISION = '752cd650bf5f649f2c0139ea02ac5627e9e60a5a'

interface PinnedAsset {
  id: string
  path: string
  bytes: number
  sha256: string
  role: string
}

// Bytes from publisher model card and pinned repository tree; digests from SHA256SUMS.
const assets = {
  tokenizer: { id: 'tokenizer', path: 'tokenizer.json', bytes: 12_217_872, sha256: 'f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b', role: 'tokenizer' },
  slow: { id: 'slow-ar', path: 'slow_ar_int8.tflite', bytes: 558_303_696, sha256: '8a54c2a1d331e0bb9efa42c98506672ab7739a1d9a77cf54f7d383c8903d882e', role: 'slow autoregressive graph' },
  fast: { id: 'fast-ar', path: 'fast_ar_int8.tflite', bytes: 67_780_320, sha256: '0a0bc9f0a969c09f9b2af69231a677db3cb93852b2b8a04609413096b5e48ac0', role: 'fast autoregressive graph' },
  codec128: { id: 'codec-128', path: 'codec_decoder_fp16_T128.tflite', bytes: 261_633_152, sha256: 'e1f187dcf29d3caef749af2aaca68039020220c6cb491c2a780faba62f232a79', role: '128-frame decoder' },
  codec192: { id: 'codec-192', path: 'codec_decoder_fp16_T192.tflite', bytes: 261_829_760, sha256: '21c04bd04ed27c42ec6fc92cf26a854850fc7c8d746fb766cca04bc3640d1b68', role: '192-frame decoder' },
  encoder: { id: 'voice-encoder', path: 'codec_encoder_fp16_10s.tflite', bytes: 419_269_280, sha256: 'e27deb6663e25919d2149c63ff70e10418f0f341df963d2ae458d1466359a977', role: '10-second voice registration' },
  voiceEnCodes: { id: 'voice-en-codes', path: 'voices/en_librispeech_1272/codes.npy', bytes: 3188, sha256: 'fb61a206bd78ee559702d9b1bab8f6f15d1a1e4ce60985a2e6922e56b6b9ffb7', role: 'English voice codes' },
  voiceEnMeta: { id: 'voice-en-meta', path: 'voices/en_librispeech_1272/meta.json', bytes: 428, sha256: '37912431a5ff36b97645167950d045b93808119b9f6b8d3cb89dbd7f30700b3f', role: 'English voice transcript and metadata' },
  voiceJaCodes: { id: 'voice-ja-codes', path: 'voices/ja_funasr_example/codes.npy', bytes: 3248, sha256: '5860590236e08def6f65cf1ea5a8451e263e0a0b9d99bd761a6933eb0a2cb922', role: 'Japanese voice codes' },
  voiceJaMeta: { id: 'voice-ja-meta', path: 'voices/ja_funasr_example/meta.json', bytes: 446, sha256: '6b3991a6665001c113e7bc1b8ae45560c63c7834806bafa148203433d634f765', role: 'Japanese voice transcript and metadata' },
} as const satisfies Record<string, PinnedAsset>

export const audio8Profiles = {
  synthesis: { id: 'synthesis', name: 'Bundled voices', includeVoiceEncoder: false, includeExtendedDecoder: false },
  extendedDecode: { id: 'extendedDecode', name: 'Extended decoder', includeVoiceEncoder: false, includeExtendedDecoder: true },
  voiceRegistration: { id: 'voiceRegistration', name: 'Voice registration', includeVoiceEncoder: true, includeExtendedDecoder: false },
} as const

export type Audio8ProfileId = keyof typeof audio8Profiles
export type Audio8Profile = (typeof audio8Profiles)[Audio8ProfileId]

export function createAudio8Manifest(profile: Audio8Profile = audio8Profiles.synthesis): ModelManifest {
  const selected: ModelAsset[] = [
    assets.tokenizer, assets.slow, assets.fast, assets.codec128,
    assets.voiceEnCodes, assets.voiceEnMeta, assets.voiceJaCodes, assets.voiceJaMeta,
  ]
  if (profile.includeExtendedDecoder) selected.push(assets.codec192)
  if (profile.includeVoiceEncoder) selected.push(assets.encoder)
  const downloadBytes = selected.reduce((sum, asset) => sum + (asset.bytes ?? 0), 0)

  return {
    modelId: 'audio8-tts-realistic-air-' + profile.id,
    name: 'Audio8 Realistic Air · ' + profile.name,
    version: '0.1.0',
    capabilities: ['text-to-speech'],
    // This is a candidate. LiteRT 2.2 Android evidence does NOT qualify LiteRT.js.
    backends: { wasm: 'experimental', webgpu: 'experimental' },
    memory: { downloadBytes, residentBytes: 2_500_000_000 },
    assets: selected,
    verification: {
      assets: 'untested', compile: 'untested', inference: 'untested',
      output: 'untested', qualification: 'unverified',
      upstreamRevision: AUDIO8_SOURCE_REVISION,
    },
  }
}
