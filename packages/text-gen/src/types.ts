export interface TextMessage {
  role: 'system' | 'user' | 'assistant' | 'model';
  content: string;
}

export interface TextGenerationInput {
  systemPrompt?: string;
  messages: TextMessage[];
}

export interface TextGenerationConfig {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  device?: 'webgpu' | 'wasm';
  dtype?: string;
  modelId?: string;
  /** Experimental tokenizer-constrained JSON Schema (single sequence). */
  jsonSchema?: Record<string,unknown>;
}

export const TEXT_GENERATION_DEFAULTS: Required<Pick<TextGenerationConfig, 'maxTokens' | 'temperature'>> = {
  maxTokens: 2048,
  temperature: 0.7,
};
