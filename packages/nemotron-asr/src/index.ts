/** Reusable, opt-in browser ONNX speech recognition; no assets load on import. */
export {
  MODEL_REVISION, MODEL_ROOT, chooseNemotronBackend, validateAudioSamples,
  type NemotronBackend, type NemotronAssets,
} from './model'
export {
  NemotronClient,
  type ChunkProgress, type StreamProgress, type NemotronClientOptions,
} from './nemotron-client'
export {
  CHUNK_SAMPLES, SAMPLE_RATE, decodeAudioFile, downmixToMono,
  logMelSpectrogram, type MelSpectrogram,
} from './audio'
