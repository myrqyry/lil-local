import { useState } from 'react'
import ModelRunner from './components/ModelRunner'
import { Qwen3TtsPanel } from './components/Qwen3TtsPanel'
import { LfmPipelinePanel } from './components/LfmPipelinePanel'
import { D1DecisionPanel } from './components/D1DecisionPanel'
import { Pix2PixPanel } from './components/Pix2PixPanel'
import { NemotronAsrPanel } from './components/NemotronAsrPanel'
import { ParakeetAsrPanel } from './components/ParakeetAsrPanel'
import { MediaPipeAudioClassifierPanel } from './components/MediaPipeAudioClassifierPanel'
import { MediaPipeVisionPanel } from './components/MediaPipeVisionPanel'
import { registeredAdapters } from './adapters/registry'
import type { ModelAdapter } from './adapters/types'

const ttsEntry: ModelAdapter = {
  modelId: 'qwen3-tts',
  metadata: { name: 'Qwen3-TTS (Speech Synthesis)', description: 'Multilingual TTS via 3-graph pipeline', modelPath: '', tags: ['tts', 'pipeline'] },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

const lfmEntry: ModelAdapter = {
  modelId: 'lfm2.5-pipelines',
  metadata: { name: 'Gemma 4 + LFM2.5 Pipelines', description: 'Gemma 4 and LFM2.5 text-gen, ColBERT retrieval, and encoders via shared pipelines', modelPath: '', tags: ['llm', 'retrieval', 'encoder', 'pipeline'] },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

const d1Entry: ModelAdapter = {
  modelId: 'd1-omni-600m-onnx-webgpu',
  metadata: {
    name: 'D1 Omni 600M (WebGPU Decisions)',
    description: 'Opt-in Q8 ONNX text-only System One decisions using ONNX Runtime Web. Browser qualification pending.',
    modelPath: '',
    tags: ['language', 'decisions', 'onnx', 'webgpu', 'pipeline'],
  },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

const pix2pixEntry: ModelAdapter = {
  modelId: 'pix2pix-edges2shoes-webgpu',
  metadata: {
    name: 'Pix2Pix · edges2shoes (Local WebGPU)',
    description: 'Choose verified folded weights from disk and run the official edges2shoes U-Net through the custom Rust/Wasm WebGPU runtime.',
    modelPath: '',
    tags: ['image', 'image-edit', 'pix2pix', 'webgpu', 'pipeline', 'local-file'],
  },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

const nemotronAsrEntry: ModelAdapter = {
  modelId: 'nemotron-3.5-asr-onnx-int4',
  metadata: { name: 'Nemotron 3.5 · Browser ASR (ONNX INT4)', description: 'Experimental local browser transcription · WebGPU or WASM · explicit download; microphone & audio files', modelPath: '', tags: ['audio', 'asr', 'onnx', 'webgpu', 'pipeline'] },
  inputSpecs: [], outputSpecs: [], isPipeline: true,
  prepareInputs: () => ({}), parseOutputs: async () => ({}),
}

const yamNetEntry: ModelAdapter = {
  modelId: 'yamnet-mediapipe',
  metadata: {
    name: 'YAMNet · MediaPipe Audio Classification',
    description: 'Opt-in local 521-class sound recognition; publisher model SHA-256 checked, 4.1 MB',
    modelPath: '',
    tags: ['audio', 'classification', 'mediapipe', 'wasm', 'pipeline'],
  },
  inputSpecs: [], outputSpecs: [], isPipeline: true,
  prepareInputs: () => ({}), parseOutputs: async () => ({}),
}

const parakeetAsrEntry: ModelAdapter = {
  modelId: 'parakeet-tdt-0.6b-mediapipe',
  metadata: {
    name: 'Parakeet TDT · MediaPipe Audio Transcriber',
    description: 'Opt-in local browser speech recognition · audio files and microphone · ~615 MB LiteRT-LM',
    modelPath: '',
    tags: ['audio', 'asr', 'mediapipe', 'wasm', 'pipeline'],
  },
  inputSpecs: [], outputSpecs: [], isPipeline: true,
  prepareInputs: () => ({}), parseOutputs: async () => ({}),
}

const mediaPipeVisionEntry: ModelAdapter = {
  modelId: 'mediapipe-vision-local',
  metadata: {
    name: 'MediaPipe Vision · Face & Gestures',
    description: 'Reusable local browser face landmarks, blendshapes and hand gestures · opt-in, verified model assets',
    modelPath: '',
    tags: ['vision', 'face', 'gesture', 'mediapipe', 'wasm', 'pipeline'],
  },
  inputSpecs: [], outputSpecs: [], isPipeline: true,
  prepareInputs: () => ({}), parseOutputs: async () => ({}),
}

const bonsaiCandidate: ModelAdapter = {
  modelId: 'bonsai2-2b-onnx-webgpu-candidate',
  metadata: {
    name: 'Bonsai 2 · 2B',
    description: 'Disabled: ONNX decoder requires unsupported Scan; native GGUF generation remains unqualified.',
    modelPath: '',
    tags: ['language', 'experimental', 'onnx', 'webgpu', 'qualification'],
  },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  disabled: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  if (selectedId === 'mediapipe-vision-local') {
    return <div className="min-h-screen bg-surface-dim"><div className="mx-auto max-w-2xl space-y-4 p-4">
      <button className="text-sm text-secondary" onClick={() => setSelectedId(null)}>← Back to model list</button>
      <h1 className="text-2xl font-bold text-on-surface">MediaPipe Vision · Local Tracking</h1>
      <MediaPipeVisionPanel />
    </div></div>
  }

  if (selectedId === 'yamnet-mediapipe') {
    return <div className="min-h-screen bg-surface-dim"><div className="mx-auto max-w-2xl space-y-4 p-4">
      <button className="text-sm text-secondary" onClick={() => setSelectedId(null)}>← Back to model list</button>
      <h1 className="text-2xl font-bold text-on-surface">MediaPipe Audio · YAMNet</h1>
      <MediaPipeAudioClassifierPanel />
    </div></div>
  }

  if (selectedId === 'parakeet-tdt-0.6b-mediapipe') {
    return <div className="min-h-screen bg-surface-dim"><div className="mx-auto max-w-2xl space-y-4 p-4">
      <button className="text-sm text-secondary" onClick={() => setSelectedId(null)}>← Back to model list</button>
      <h1 className="text-2xl font-bold text-on-surface">Parakeet TDT · Local Speech Recognition</h1>
      <ParakeetAsrPanel />
    </div></div>
  }

  if (selectedId === 'nemotron-3.5-asr-onnx-int4') {
    return <div className="min-h-screen bg-surface-dim"><div className="mx-auto max-w-2xl space-y-4 p-4">
      <button className="text-sm text-secondary" onClick={() => setSelectedId(null)}>← Back to model list</button>
      <h1 className="text-2xl font-bold text-on-surface">Nemotron 3.5 · Local Speech Recognition</h1>
      <NemotronAsrPanel />
    </div></div>
  }

  if (selectedId === 'qwen3-tts') {
    return (
      <div className="min-h-screen bg-surface-dim">
        <div className="mx-auto max-w-2xl p-4">
          <button
            className="mb-4 text-sm text-secondary transition-colors hover:text-on-secondary-container"
            onClick={() => setSelectedId(null)}
          >
            ← Back to model list
          </button>
          <h1 className="mb-4 text-2xl font-bold text-on-surface">Qwen3-TTS</h1>
          <Qwen3TtsPanel />
        </div>
      </div>
    )
  }

  if (selectedId === 'd1-omni-600m-onnx-webgpu') {
    return (
      <div className="min-h-screen bg-surface-dim">
        <div className="mx-auto max-w-2xl p-4">
          <button
            className="mb-4 text-sm text-secondary transition-colors hover:text-on-secondary-container"
            onClick={() => setSelectedId(null)}
          >← Back to model list</button>
          <h1 className="mb-4 text-2xl font-bold text-on-surface">D1 Omni · WebGPU</h1>
          <D1DecisionPanel />
        </div>
      </div>
    )
  }

  if (selectedId === 'pix2pix-edges2shoes-webgpu') {
    return (
      <div className="min-h-screen bg-surface-dim">
        <div className="mx-auto max-w-3xl p-4">
          <button
            className="mb-4 text-sm text-secondary transition-colors hover:text-on-secondary-container"
            onClick={() => setSelectedId(null)}
          >← Back to model list</button>
          <h1 className="mb-4 text-2xl font-bold text-on-surface">Pix2Pix · Local WebGPU</h1>
          <Pix2PixPanel />
        </div>
      </div>
    )
  }

  if (selectedId === 'lfm2.5-pipelines') {
    return (
      <div className="min-h-screen bg-surface-dim">
        <div className="mx-auto max-w-2xl p-4">
          <button
            className="mb-4 text-sm text-secondary transition-colors hover:text-on-secondary-container"
            onClick={() => setSelectedId(null)}
          >
            ← Back to model list
          </button>
          <h1 className="mb-4 text-2xl font-bold text-on-surface">Gemma 4 + LFM2.5 Pipelines</h1>
          <LfmPipelinePanel />
        </div>
      </div>
    )
  }

  return (
    <ModelRunner
      adapters={[...registeredAdapters, ttsEntry, lfmEntry, d1Entry, pix2pixEntry, nemotronAsrEntry, parakeetAsrEntry, yamNetEntry, mediaPipeVisionEntry, bonsaiCandidate]}
      selectedId={selectedId}
      onSelect={setSelectedId}
    />
  )
}
