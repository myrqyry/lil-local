import type {
  ConversationConfig,
  Message,
  SamplerParameters,
  SessionConfig,
} from '@litert-lm/core';
import { MediaPipeMultimodalEngine } from './mediapipe-engine';
import type {
  LiteRtLmWorkerEngine,
  LiteRtLmWorkerGenerationConfig,
  LiteRtLmWorkerMessage,
  LiteRtLmWorkerPrompt,
  LiteRtLmWorkerRequest,
  LiteRtLmWorkerResponse,
} from './protocol';

interface WorkerScope {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown): void;
}

const LITERT_LM_0190_WASM_BASE = 'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.19.0/wasm';

interface LiteRtLmModule {
  getOrLoadGlobalLiteRtLm(path: string): Promise<unknown>;
  Engine: {
    create: (settings: {
      model: string | Blob | ReadableStream<Uint8Array>;
      visionBackend?: number;
      audioBackend?: number;
      mainExecutorSettings?: { maxNumTokens?: number; maxNumImages?: number };
    }) => Promise<LiteRtLmEngine>;
  };
}

interface LiteRtLmEngine {
  createConversation(config?: ConversationConfig): Promise<LiteRtLmConversation>;
  delete(): Promise<void>;
}

interface LiteRtLmConversation {
  sendMessage(message: Message): Promise<{ text?: string }>;
  sendMessageStreaming(message: Message): ReadableStream<Message>;
  cancel(): void;
  delete(): Promise<void>;
  getTokenCount?(): Promise<number>;
  getBenchmarkInfo?(): Promise<{
    lastPrefillTokensPerSecond: number;
    lastDecodeTokensPerSecond: number;
  }>;
}

type Conversation = Awaited<ReturnType<LiteRtLmEngine['createConversation']>>;

const worker = self as unknown as WorkerScope;

let liteRtLmEngine: LiteRtLmEngine | undefined;
let mediaPipeEngine: MediaPipeMultimodalEngine | undefined;
let activeEngine: LiteRtLmWorkerEngine = 'litert-lm';
let liteRtLmVisionEnabled = false;
let liteRtLmAudioEnabled = false;
let dispatchQueue: Promise<void> = Promise.resolve();
const activeConversations = new Map<string, Conversation>();
// At most one native chat stays alive per worker. Unlike request IDs, its
// identity spans multiple turns without replaying/prefilling past messages.
let retainedChat: { sessionId: string; configKey: string; conversation: Conversation; invalid: boolean } | undefined;
const mediaPipeGenerationIds = new Set<string>();
const knownGenerationIds = new Set<string>();
const cancelledGenerations = new Set<string>();

function extractText(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter((part) => part.type === 'text' && typeof part.text === 'string')
      .map((part) => (part as { text: string }).text)
      .join('');
  }
  return '';
}

function extractReasoning(message: Message): string {
  return message.channels?.reasoning ?? message.channels?.thought ?? message.channels?.think ?? '';
}

function emit(response: LiteRtLmWorkerResponse): void {
  worker.postMessage(response);
}

function buildConversationConfig(config?: LiteRtLmWorkerGenerationConfig): ConversationConfig | undefined {
  const conversationConfig: ConversationConfig = {};
  const hasSampler =
    config?.temperature !== undefined ||
    config?.topK !== undefined ||
    config?.topP !== undefined ||
    config?.seed !== undefined;
  if (hasSampler || config?.maxOutputTokens !== undefined) {
    const sessionConfig: SessionConfig = {};
    if (hasSampler) {
      const samplerParams: SamplerParameters = {};
      if (config?.temperature !== undefined) samplerParams.temperature = config.temperature;
      if (config?.topK !== undefined) samplerParams.k = config.topK;
      if (config?.topP !== undefined) samplerParams.p = config.topP;
      if (config?.seed !== undefined) samplerParams.seed = config.seed;
      sessionConfig.samplerParams = samplerParams;
    }
    if (config?.maxOutputTokens !== undefined) sessionConfig.maxOutputTokens = config.maxOutputTokens;
    conversationConfig.sessionConfig = sessionConfig;
  }
  if (config?.systemPrompt?.trim()) {
    conversationConfig.preface = { messages: [{ role: 'system', content: config.systemPrompt }] };
  }
  if (config?.enableThinking !== undefined) {
    conversationConfig.preface ??= {};
    conversationConfig.preface.extra_context = {enable_thinking: config.enableThinking};
  }
  return Object.keys(conversationConfig).length > 0 ? conversationConfig : undefined;
}

async function replayHistory(conversation: Conversation, history?: LiteRtLmWorkerMessage[]): Promise<void> {
  if (!history?.length) return;
  for (const message of history) {
    await conversation.sendMessage({
      role: message.role === 'assistant' ? 'model' : message.role,
      content: message.content,
    });
  }
}

async function streamResponse(
  stream: ReadableStream<Message>,
  onMessage: (message: Message) => void,
): Promise<void> {
  if (Symbol.asyncIterator in stream) {
    for await (const chunk of stream) onMessage(chunk);
  } else {
    const reader = (stream as ReadableStream<Message>).getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        onMessage(value);
      }
    } finally {
      reader.releaseLock();
    }
  }
}

async function disposeConversation(id: string, conversation: Conversation): Promise<void> {
  if (activeConversations.get(id) !== conversation) return;
  await conversation.delete();
  if (activeConversations.get(id) === conversation) {
    activeConversations.delete(id);
  }
}

function toLiteRtLmContent(prompt: LiteRtLmWorkerPrompt): Message['content'] {
  if (typeof prompt === 'string') return prompt;
  if (prompt.some((part) => part.type === 'audio') && !liteRtLmAudioEnabled) {
    throw new Error('Audio input requires audioBackend when loading the LiteRT-LM engine');
  }
  if (prompt.some((part) => part.type === 'audio' && !('data' in part))) {
    throw new Error('LiteRT-LM requires a decoded WAV audio Blob; PCM arrays belong to the MediaPipe backend');
  }
  if (prompt.some((part) => part.type === 'image') && !liteRtLmVisionEnabled) {
    throw new Error('Image input requires visionBackend when loading the LiteRT-LM engine');
  }
  return prompt.map((part) => {
    if (part.type === 'text') return { type: 'text' as const, text: part.text };
    if (part.type === 'image') return { type: 'image' as const, data: part.data };
    if (!('data' in part)) throw new Error('LiteRT-LM requires WAV Blob audio');
    return { type: 'audio' as const, data: part.data };
  });
}

async function generateLiteRtLm(
  id: string,
  prompt: LiteRtLmWorkerPrompt,
  config?: LiteRtLmWorkerGenerationConfig,
): Promise<void> {
  if (cancelledGenerations.has(id)) return;
  if (!liteRtLmEngine) throw new Error('LiteRT-LM is not loaded');
  const sessionId = config?.sessionId;
  if (sessionId !== undefined && (!/^[a-zA-Z0-9_-]{1,100}$/.test(sessionId))) {
    throw new Error('sessionId must be 1–100 letters, digits, underscores or hyphens');
  }
  if (sessionId && config?.history?.length) {
    throw new Error('Native chat sessions cannot also replay history');
  }
  const {sessionId: _sessionId, history: _history, ...sessionOptions} = config ?? {};
  const configKey = JSON.stringify(sessionOptions);
  if (retainedChat && (!sessionId || retainedChat.sessionId !== sessionId || retainedChat.invalid)) {
    await retainedChat.conversation.delete();
    retainedChat = undefined;
  }
  if (retainedChat && retainedChat.configKey !== configKey) {
    throw new Error('Native chat settings changed; start a new chat session');
  }
  const conversation = retainedChat?.conversation
    ?? await liteRtLmEngine.createConversation(buildConversationConfig(config));
  if (cancelledGenerations.has(id)) {
    if (retainedChat?.conversation === conversation) retainedChat.invalid = true;
    else await conversation.delete();
    return;
  }
  if (sessionId && !retainedChat) {
    retainedChat = {sessionId, configKey, conversation, invalid: false};
  }
  activeConversations.set(id, conversation);
  let generationFailed = false;
  let generationError: unknown;
  try {
    await replayHistory(conversation, config?.history);
    await streamResponse(
      conversation.sendMessageStreaming({ role: 'user', content: toLiteRtLmContent(prompt) }),
      (message) => {
        const reasoning = extractReasoning(message);
        if (reasoning) emit({ type: 'reasoning', id, text: reasoning });
        const text = extractText(message);
        if (text) emit({ type: 'token', id, text });
      },
    );
    // Native performance data is advisory; a failed stats query must never
    // erase an otherwise successful generation. Only multi-turn sessions
    // expose this telemetry to the playground.
    if (sessionId && !cancelledGenerations.has(id)) {
      try {
        const tokens = await conversation.getTokenCount?.();
        const benchmark = await conversation.getBenchmarkInfo?.();
        if (typeof tokens === 'number' && Number.isFinite(tokens) &&
            benchmark && Number.isFinite(benchmark.lastPrefillTokensPerSecond) &&
            Number.isFinite(benchmark.lastDecodeTokensPerSecond)) {
          emit({
            type: 'stats', id,
            data: {
              tokenCount: tokens,
              prefillTokensPerSecond: benchmark.lastPrefillTokensPerSecond,
              decodeTokensPerSecond: benchmark.lastDecodeTokensPerSecond,
            },
          });
        }
      } catch (error) {
        console.warn('[text-gen worker] Native conversation stats unavailable', error);
      }
    }
  } catch (error) {
    generationFailed = true;
    generationError = error;
  }

  // Successful native chat turns keep their KV cache, but failures and
  // cancellation invalidate the session, so a future prompt cannot inherit
  // a half-written state.
  const keepConversation = Boolean(sessionId && !generationFailed && !cancelledGenerations.has(id));
  if (!keepConversation && retainedChat?.conversation === conversation) {
    retainedChat.invalid = true;
  }
  try {
    if (keepConversation) {
      activeConversations.delete(id);
    } else {
      await disposeConversation(id, conversation);
      if (retainedChat?.conversation === conversation) retainedChat = undefined;
    }
  } catch (cleanupError) {
    let attachedToPrimary = false;
    if (generationFailed && generationError !== null) {
      const kind = typeof generationError;
      if (kind === 'object' || kind === 'function') {
        try {
          const errorWithCause = generationError as { cause?: unknown };
          if (errorWithCause.cause === undefined) {
            errorWithCause.cause = cleanupError;
            attachedToPrimary = errorWithCause.cause === cleanupError;
          }
        } catch {
          // Frozen/cross-realm throwables may reject mutation; warn below.
        }
      }
    }

    console.warn(
      '[text-gen worker] LiteRT-LM conversation cleanup failed',
      cleanupError,
      attachedToPrimary ? '(attached to primary error)' : '(not attached to primary error)',
    );
  }

  if (generationFailed) throw generationError;

  // Cleanup settles before the terminal event so cancellation can still win
  // during teardown, but cleanup failure cannot erase successful inference.
  if (!cancelledGenerations.has(id)) {
    emit({ type: 'complete', id });
  }
}

async function generateMediaPipe(id: string, prompt: LiteRtLmWorkerPrompt): Promise<void> {
  if (!mediaPipeEngine) throw new Error('MediaPipe LLM Inference is not loaded');
  if (cancelledGenerations.has(id)) return;

  mediaPipeGenerationIds.add(id);
  try {
    await mediaPipeEngine.generate(
      prompt,
      (text) => {
        if (!cancelledGenerations.has(id) && text) {
          emit({ type: 'token', id, text });
        }
      },
      () => cancelledGenerations.has(id),
    );
    if (!cancelledGenerations.has(id)) {
      emit({ type: 'complete', id });
    }
  } catch (error) {
    // A caller-side abort already rejected the public promise and asked the
    // engine to cancel. Do not turn that expected cancellation into a second
    // worker error after the request has been abandoned.
    if (!cancelledGenerations.has(id)) throw error;
  } finally {
    // Dispatcher owns cancelledGenerations so it can emit one terminal
    // cancellation event before clearing request bookkeeping.
    mediaPipeGenerationIds.delete(id);
  }
}

async function disposeAllConversations(): Promise<void> {
  const disposed = new Set<Conversation>();
  for (const [id, conversation] of activeConversations) {
    conversation.cancel();
    if (!disposed.has(conversation)) {
      await conversation.delete();
      disposed.add(conversation);
    }
    activeConversations.delete(id);
  }
  if (retainedChat) {
    if (!disposed.has(retainedChat.conversation)) await retainedChat.conversation.delete();
    retainedChat = undefined;
  }
}

async function disposeLoadedEngine(): Promise<void> {
  await disposeAllConversations();

  if (liteRtLmEngine) {
    await liteRtLmEngine.delete();
    liteRtLmEngine = undefined;
  }

  if (mediaPipeEngine) {
    mediaPipeEngine.dispose();
    mediaPipeEngine = undefined;
  }
}

async function loadEngine(data: Extract<LiteRtLmWorkerRequest, { type: 'load' }>): Promise<void> {
  // Real Chromium tests with the official E2B web file returned a null
  // vision executor and a WASM abort for audio. Reject those exact
  // browser checkpoint variants before initializing any native engine.
  // Other (native) model files retain their explicit multimodal path.
  const selectedModel = typeof data.model === 'string'
    ? data.model
    : (data.model as Blob & { name?: string }).name;
  if ((data.options?.engine ?? 'litert-lm') === 'litert-lm' &&
      (data.options?.visionBackend || data.options?.audioBackend) &&
      typeof selectedModel === 'string' &&
      /(?:^|\/)gemma-4-E(?:2|4)B-it-web\.litertlm(?:$|[?#])/i.test(selectedModel)) {
    throw new Error('Gemma 4 web checkpoints are text-only in LiteRT-LM JS 0.19.0; real browser image/audio executors failed. Use a compatible native runtime for multimodal input.');
  }
  await disposeLoadedEngine();
  liteRtLmVisionEnabled = false;
  liteRtLmAudioEnabled = false;

  activeEngine = data.options?.engine ?? 'litert-lm';
  if (activeEngine === 'mediapipe') {
    mediaPipeEngine = new MediaPipeMultimodalEngine();
    await mediaPipeEngine.load(data.model, data.options?.mediaPipe);
    return;
  }

  const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
  // Pin 0.19.0 WASM explicitly rather than relying on SDK defaults.
  // Explicit loading also detects conflicts with a previously loaded runtime.
  const wasmBase = data.options?.wasmBase ?? LITERT_LM_0190_WASM_BASE;
  // LiteRT-LM's Emscripten ModuleFactory resolves its .wasm name against
  // self.location (the worker asset), not the URL passed to importScripts().
  // Supply a trusted, absolute locateFile mapping before initializing Wasm.
  const runtimeBase = new URL(
    wasmBase.endsWith('/') ? wasmBase : wasmBase + '/',
    (worker as WorkerScope & { location?: { href: string } }).location?.href ?? 'http://localhost/',
  );
  (worker as WorkerScope & { Module?: { locateFile: (file: string) => string } }).Module = {
    locateFile: (file: string) => new URL(file, runtimeBase).href,
  };
  await module.getOrLoadGlobalLiteRtLm(wasmBase);
  const maxNumImages = data.options?.maxNumImages;
  if (maxNumImages !== undefined &&
      (!Number.isSafeInteger(maxNumImages) || maxNumImages < 0 || maxNumImages > 8)) {
    throw new Error('maxNumImages must be an integer from 0 through 8');
  }
  liteRtLmEngine = await module.Engine.create({
    model: data.model,
    ...(data.options?.visionBackend ? { visionBackend: data.options.visionBackend === 'gpu' ? 4 : 3 } : {}),
    ...(data.options?.audioBackend ? { audioBackend: data.options.audioBackend === 'gpu' ? 4 : 3 } : {}),
    mainExecutorSettings: {
      maxNumTokens: 8192,
      ...(maxNumImages !== undefined ? { maxNumImages } : {}),
    },
  });
  liteRtLmVisionEnabled = Boolean(data.options?.visionBackend);
  liteRtLmAudioEnabled = Boolean(data.options?.audioBackend);
}

function hasGenerationConfig(config?: LiteRtLmWorkerGenerationConfig): boolean {
  return config !== undefined && Object.keys(config).length > 0;
}

function cancelGeneration(id: string): void {
  // Ignore late/unknown cancels so cancellation bookkeeping cannot grow without
  // a corresponding request that will eventually clean it up.
  if (!knownGenerationIds.has(id)) return;

  cancelledGenerations.add(id);
  activeConversations.get(id)?.cancel();
  if (mediaPipeGenerationIds.has(id)) {
    mediaPipeEngine?.cancel();
  }
}

async function dispatchRequest(data: Exclude<LiteRtLmWorkerRequest, { type: 'cancel' }>): Promise<void> {
  try {
    switch (data.type) {
      case 'load':
        await loadEngine(data);
        emit({ type: 'ready' });
        break;
      case 'generate':
        try {
          if (cancelledGenerations.has(data.id)) return;

          if (activeEngine === 'mediapipe') {
            if (hasGenerationConfig(data.config)) {
              throw new Error(
                'Per-generation config is not supported by the MediaPipe worker engine; set MediaPipe sampling/context options during load() instead',
              );
            }
            await generateMediaPipe(data.id, data.prompt);
          } else {
            await generateLiteRtLm(data.id, data.prompt, data.config);
          }
        } catch (error) {
          // Cancellation has its own terminal worker response. Suppress native
          // cancellation errors here and let finally emit the cancelled event.
          if (!cancelledGenerations.has(data.id)) throw error;
        } finally {
          const wasCancelled = cancelledGenerations.has(data.id);
          knownGenerationIds.delete(data.id);
          cancelledGenerations.delete(data.id);
          if (wasCancelled) emit({ type: 'cancelled', id: data.id });
        }
        break;
      case 'dispose':
        await disposeLoadedEngine();
        emit({ type: 'disposed' });
        break;
    }
  } catch (error) {
    emit({
      type: 'error',
      id: data.type === 'generate' ? data.id : undefined,
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

worker.onmessage = (event: MessageEvent<LiteRtLmWorkerRequest>) => {
  const data = event.data;

  // Cancellation is deliberately outside the serialized dispatch queue so an
  // active decode can be interrupted immediately.
  if (data.type === 'cancel') {
    cancelGeneration(data.id);
    return;
  }

  if (data.type === 'dispose') {
    // Teardown itself remains serialized, but interrupt native work now so the
    // queued dispose is not forced to wait for an abandoned generation.
    for (const id of knownGenerationIds) cancelGeneration(id);
  }

  if (data.type === 'generate') {
    knownGenerationIds.add(data.id);
  }

  const task = dispatchQueue
    .catch(() => undefined)
    .then(() => dispatchRequest(data));
  dispatchQueue = task.then(
    () => undefined,
    () => undefined,
  );
};
