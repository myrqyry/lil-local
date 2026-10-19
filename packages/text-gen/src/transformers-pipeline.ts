import {
  type Pipeline,
  type PipelineProgress,
  type PipelineStatus,
  type RuntimeContext,
  type TextInferenceResult,
} from '@lil-local/inference-core';
import {
  TEXT_GENERATION_DEFAULTS,
  type TextGenerationConfig,
  type TextGenerationInput,
  type TextMessage,
} from './types';
import { transformersTextManifest } from './manifest';

interface TransformersModule {
  pipeline: (
    task: string,
    modelId: string,
    options?: {
      device?: string;
      dtype?: string;
      progress_callback?: (p: { progress?: number }) => void;
    },
  ) => Promise<TransformersGenerator>;
}

interface TransformersGenerator {
  (
    messages: Array<{ role: string; content: string }>,
    options: Record<string, unknown>,
  ): Promise<{ output: Array<{ generated_text?: string } | { content?: string }> }>;
}

export interface TransformersTextConfig extends TextGenerationConfig {
  modelId: string;
  device: 'webgpu' | 'wasm';
  dtype?: string;
}

const DEFAULTS: TransformersTextConfig = {
  modelId: 'onnx-community/Qwen3-0.6B-ONNX',
  device: 'wasm',
  dtype: 'q4f16',
};

function toChatMessages(input: TextGenerationInput): Array<{ role: string; content: string }> {
  const messages: Array<{ role: string; content: string }> = [];
  if (input.systemPrompt) {
    messages.push({ role: 'system', content: input.systemPrompt });
  }
  for (const msg of input.messages) {
    const role = msg.role === 'model' ? 'assistant' : msg.role;
    messages.push({ role, content: msg.content });
  }
  return messages;
}

export class TransformersTextPipeline
  implements Pipeline<TextGenerationInput, TextInferenceResult, TransformersTextConfig>
{
  readonly manifest = transformersTextManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
  private generator: TransformersGenerator | null = null;
  private loadMs = 0;

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    this.status = 'loading';
    this.context = context;
    const loadStart = performance.now();
    try {
      this.report({ phase: 'loading', step: 1, total: 2 });
      const transformers = (await import('@huggingface/transformers')) as unknown as TransformersModule;
      this.report({ phase: 'loading', step: 2, total: 2 });
      this.generator = await transformers.pipeline('text-generation', this.manifest.assets[0].path, {
        device: DEFAULTS.device,
        dtype: DEFAULTS.dtype,
        progress_callback: () => {},
      });
      this.loadMs = performance.now() - loadStart;
      this.status = 'ready';
    } catch (e) {
      this.status = 'error';
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async run(
    input: TextGenerationInput,
    config?: TransformersTextConfig,
    signal?: AbortSignal,
  ): Promise<TextInferenceResult> {
    if (this.status !== 'ready') throw new Error('Pipeline not ready');
    if (!this.generator) throw new Error('Transformers text pipeline not loaded');
    this.status = 'running';
    const cfg = { ...DEFAULTS, ...config };
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const messages = toChatMessages(input);
      const result = await this.generator(messages, {
        max_new_tokens: cfg.maxTokens ?? TEXT_GENERATION_DEFAULTS.maxTokens,
        do_sample: (cfg.temperature ?? TEXT_GENERATION_DEFAULTS.temperature) > 0,
        temperature: cfg.temperature,
        top_p: cfg.topP,
        top_k: cfg.topK,
      });
      if (signal?.aborted) throw new Error('CANCELLED');
      const first = result.output[0];
      const text =
        first && 'generated_text' in first && first.generated_text
          ? first.generated_text
          : first && 'content' in first && first.content
            ? first.content
            : '';
      this.status = 'ready';
      return {
        kind: 'text',
        text,
      } satisfies TextInferenceResult;
    } catch (e) {
      this.status = 'ready';
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async dispose(): Promise<void> {
    this.generator = null;
    this.context = null;
    this.status = 'disposed';
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
