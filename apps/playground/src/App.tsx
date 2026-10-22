import { useState } from 'react'
import ModelRunner from './components/ModelRunner'
import { Qwen3TtsPanel } from './components/Qwen3TtsPanel'
import { LfmPipelinePanel } from './components/LfmPipelinePanel'
import { D1DecisionPanel } from './components/D1DecisionPanel'
import { Bonsai2CandidatePanel } from './components/Bonsai2CandidatePanel'
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

const bonsaiCandidate: ModelAdapter = {
  modelId: 'bonsai2-2b-onnx-webgpu-candidate',
  metadata: {
    name: 'Bonsai 2 · 2B (ONNX audit + Native)',
    description: 'Published ONNX decoder blocked by Scan; local PrismML GGUF server optional and not yet runtime-qualified.',
    modelPath: '',
    tags: ['language', 'experimental', 'onnx', 'webgpu', 'qualification'],
  },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

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

  if (selectedId === 'bonsai2-2b-onnx-webgpu-candidate') {
    return (
      <div className="min-h-screen bg-surface-dim">
        <div className="mx-auto max-w-2xl p-4">
          <button
            className="mb-4 text-sm text-secondary transition-colors hover:text-on-secondary-container"
            onClick={() => setSelectedId(null)}
          >← Back to model list</button>
          <Bonsai2CandidatePanel />
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
      adapters={[...registeredAdapters, ttsEntry, lfmEntry, d1Entry, bonsaiCandidate]}
      selectedId={selectedId}
      onSelect={setSelectedId}
    />
  )
}
