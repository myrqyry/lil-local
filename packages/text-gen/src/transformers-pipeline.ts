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
  tokenizer?: unknown;
  dispose?: () => Promise<void> | void;
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
  private pendingLoad: Promise<void> | null = null;
  private pendingDevice: 'webgpu' | 'wasm' | null = null;
  private readyDevice: 'webgpu' | 'wasm' | null = null;
  private activeRun: Promise<TextInferenceResult> | null = null;
  private disposal: Promise<void> | null = null;
  private disposeRequested = false;

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
    if (this.disposeRequested) throw new Error('Transformers pipeline disposed');
    if (this.pendingLoad) {
      if (this.pendingDevice !== device) {
        throw new Error(`Transformers pipeline already loading on ${this.pendingDevice}; dispose before changing device`);
      }
      return this.pendingLoad;
    }
    if (this.generator) {
      if (this.readyDevice !== device) {
        throw new Error(`Transformers pipeline already loaded on ${this.readyDevice}; dispose before changing device`);
      }
      return;
    }
    this.status = 'loading';
    this.pendingDevice = device;
    const loading = (async () => {
      try {
        this.report({ phase: 'loading', step: 1, total: 2 });
        const transformers = (await import('@huggingface/transformers')) as unknown as TransformersModule;
        this.report({ phase: 'loading', step: 2, total: 2 });
        const generator = await transformers.pipeline('text-generation', this.options.modelId, {
          device,
          dtype: this.options.dtype,
          ...(this.options.revision ? {revision:this.options.revision} : {}),
          progress_callback: this.options.onProgress ?? (() => {}),
        });
        // Keep a completed late load reachable for dispose(), but never revive it.
        this.generator = generator;
        this.readyDevice = device;
        if (this.disposeRequested) throw new Error('Transformers pipeline disposed during load');
        this.status = 'ready';
      } catch (e) {
        if (!this.disposeRequested) this.status = 'error';
        throw e instanceof Error ? e : new Error(String(e));
      }
    })();
    this.pendingLoad = loading;
    try {
      await loading;
    } finally {
      if (this.pendingLoad === loading) {
        this.pendingLoad = null;
        this.pendingDevice = null;
      }
    }
  }

  run(
    input: TextGenerationInput,
    config?: Partial<TransformersTextConfig>,
    signal?: AbortSignal,
  ): Promise<TextInferenceResult> {
    if (this.disposeRequested || this.status !== 'ready') {
      return Promise.reject(new Error('Pipeline not ready'));
    }
    if (!this.generator) return Promise.reject(new Error('Transformers text pipeline not loaded'));
    this.status = 'running';
    const pending = this.generate(input, config, signal);
    this.activeRun = pending;
    return pending.finally(() => {
      if (this.activeRun === pending) this.activeRun = null;
      if (!this.disposeRequested) this.status = 'ready';
    });
  }

  private async generate(
    input: TextGenerationInput,
    config?: Partial<TransformersTextConfig>,
    signal?: AbortSignal,
  ): Promise<TextInferenceResult> {
    const cfg = { ...this.options, ...config };
    const generator = this.generator;
    if (!generator) throw new Error('Transformers text pipeline not loaded');
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const messages = toChatMessages(input);
      const constraint = cfg.jsonSchema
        ? await (async () => {
          if (!generator.tokenizer) {
            throw new Error('Constrained JSON requires the loaded model tokenizer.');
          }
          const { StructuredOutputProcessor } =
            await import('@huggingface/transformers-structured-output');
          return [new StructuredOutputProcessor(
            generator.tokenizer as ConstructorParameters<typeof StructuredOutputProcessor>[0],
            {type:'json_schema',json_schema:cfg.jsonSchema as Extract<ConstructorParameters<typeof StructuredOutputProcessor>[1], {type:'json_schema'}>['json_schema']},
          )];
        })()
        : undefined;
      const result = await generator(messages, {
        max_new_tokens: cfg.maxTokens ?? TEXT_GENERATION_DEFAULTS.maxTokens,
        do_sample: (cfg.temperature ?? TEXT_GENERATION_DEFAULTS.temperature) > 0,
        temperature: cfg.temperature,
        top_p: cfg.topP,
        top_k: cfg.topK,
        ...(constraint ? {logits_processor: constraint} : {}),
      });
      if (signal?.aborted) throw new Error('CANCELLED');
      const text = extractAssistantText(result);
      if (!text) throw new Error('Transformers pipeline did not produce assistant text');
      return {
        kind: 'text',
        text,
      } satisfies TextInferenceResult;
    } catch (e) {
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.disposeRequested = true;
    this.disposal = (async () => {
      try {
        // Model creation and generation can finish after teardown was requested.
        // Wait for those operations before releasing the provider's actual resources.
        if (this.pendingLoad) await this.pendingLoad.catch(() => {});
        if (this.activeRun) await this.activeRun.catch(() => {});
        await this.generator?.dispose?.();
      } finally {
        this.generator = null;
        this.readyDevice = null;
        this.status = 'disposed';
      }
    })();
    return this.disposal;
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
