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
      revision?: string;
      progress_callback?: (p: { progress?: number }) => void;
    },
  ) => Promise<TransformersGenerator>;
}

interface TransformersGenerator {
  (
    messages: Array<{ role: string; content: string }>,
    options: Record<string, unknown>,
  ): Promise<unknown>;
}

export interface TransformersTextConfig extends TextGenerationConfig {
  modelId: string;
  device: 'webgpu' | 'wasm';
  dtype?: string;
  revision?: string;
  onProgress?: (progress: {status?: string; progress?: number; file?: string}) => void;
}

const DEFAULTS: TransformersTextConfig = {
  modelId: 'onnx-community/Qwen3-0.6B-ONNX',
  device: 'wasm',
  dtype: 'q4',
};

function extractAssistantText(result: unknown): string {
  // Transformers.js 4 returns [{generated_text: [{role,content}, ...]}],
  // older versions can return {output:[{generated_text:string}]}.
  const value = Array.isArray(result) ? result[0]
    : result && typeof result === 'object' && 'output' in result
      ? (result as {output?: unknown[]}).output?.[0] : result;
  if (!value || typeof value !== 'object') return '';
  const generated = (value as {generated_text?: unknown;content?: unknown}).generated_text
    ?? (value as {content?:unknown}).content;
  if (typeof generated === 'string') return generated.trim();
  if (Array.isArray(generated)) {
    const assistant = [...generated].reverse().find(msg =>
      msg && typeof msg === 'object' && (msg as {role?:string}).role === 'assistant'
      && typeof (msg as {content?:unknown}).content === 'string'
    ) as {content:string}|undefined;
    return assistant?.content.trim() ?? '';
  }
  return '';
}

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

  private generator: TransformersGenerator | null = null;
  private readonly options: TransformersTextConfig;

  constructor(options: Partial<TransformersTextConfig> = {}) {
    this.options = { ...DEFAULTS, ...options };
  }

  async load(context: RuntimeContext): Promise<void> {
    return this.loadForDevice(
      context.backend === 'webgpu' || context.backend === 'wasm'
        ? context.backend : this.options.device,
    );
  }

  /** Direct browser consumer API; no unrelated LiteRT runtime services. */
  async loadForDevice(device: 'webgpu' | 'wasm'): Promise<void> {
    if (this.status === 'ready') return;
    this.status = 'loading';
    try {
      this.report({ phase: 'loading', step: 1, total: 2 });
      const transformers = (await import('@huggingface/transformers')) as unknown as TransformersModule;
      this.report({ phase: 'loading', step: 2, total: 2 });
      this.generator = await transformers.pipeline('text-generation', this.options.modelId, {
        device,
        dtype: this.options.dtype,
        ...(this.options.revision ? {revision:this.options.revision} : {}),
        progress_callback: this.options.onProgress ?? (() => {}),
      });
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
    const cfg = { ...this.options, ...config };
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
      const text = extractAssistantText(result);
      if (!text) throw new Error('Transformers pipeline did not produce assistant text');
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
    this.status = 'disposed';
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
