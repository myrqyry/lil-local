import type { BrowserSubstrate } from '@lil-local/inference-core';

/** ONNX source pin: model weights and tokenizer are fetched only by load(). */
export const PRIVACY_MODEL_ID = 'openai/privacy-filter';
export const PRIVACY_MODEL_REVISION = '7ffa9a043d54d1be65afb281eddf0ffbe629385b';

export const PRIVACY_CATEGORIES = [
  'account_number', 'private_address', 'private_date', 'private_email',
  'private_person', 'private_phone', 'private_url', 'secret',
] as const;
export type PrivacyCategory = typeof PRIVACY_CATEGORIES[number];
export type PrivacyDtype = 'q4' | 'q4f16' | 'fp16' | 'fp32' | 'q8';
export type PrivacyDevice = 'webgpu' | 'wasm';

export interface PrivacySpan {
  category: PrivacyCategory;
  start: number;
  end: number;
  text: string;
  confidence: number;
}
export interface PrivacyInput { text: string }
export interface PrivacyResult {
  kind: 'privacy-detection';
  spans: PrivacySpan[];
  dtype: PrivacyDtype;
  device: PrivacyDevice;
}
export interface PrivacyFilterOptions {
  /** Caller selection overrides the "smallest compatible" default. */
  dtype?: PrivacyDtype;
  device?: PrivacyDevice;
  /** Progress goes only to the caller. Text is never logged by this package. */
  onProgress?: (progress: {status?: string; progress?: number; file?: string}) => void;
}
export interface RedactOptions { categories?: readonly PrivacyCategory[] }

export function choosePrivacyFilterVariant(
  device: PrivacyDevice, shaderF16: boolean, requested?: PrivacyDtype,
): PrivacyDtype {
  const dtype = requested ?? (device === 'webgpu' && shaderF16 ? 'q4f16' : 'q4');
  if ((dtype === 'q4f16' || dtype === 'fp16') && device === 'webgpu' && !shaderF16) {
    throw new Error(dtype + ' needs WebGPU shader-f16 support; use q4');
  }
  if (dtype === 'q4f16' && device === 'wasm') {
    throw new Error('q4f16 requires WebGPU');
  }
  return dtype;
}

type RawEntity = { entity_group: string; score: number; word: string; start?: number; end?: number };
function isPrivacyCategory(value: string): value is PrivacyCategory {
  return (PRIVACY_CATEGORIES as readonly string[]).includes(value);
}

/**
 * Transformers.js 4.3.1's token-classification pipeline omits offsets.
 * Do not guess at repetitions or normalization-induced changes: fail instead.
 * Distinct from upstream constrained-Viterbi BIOES decoding.
 */
export function resolvePrivacySpans(text: string, raw: unknown): PrivacySpan[] {
  if (typeof text !== 'string') throw new TypeError('Privacy input must be text');
  if (!Array.isArray(raw)) throw new Error('Privacy model returned non-array entities');
  const spans: PrivacySpan[] = [];
  for (const entity of raw) {
    if (!entity || typeof entity !== 'object') throw new Error('Malformed privacy entity');
    const { entity_group, score, word, start, end } = entity as RawEntity;
    if (typeof entity_group !== 'string' || !isPrivacyCategory(entity_group) ||
      typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1 ||
      typeof word !== 'string' || word.length === 0) {
      throw new Error('Invalid privacy entity label, confidence, or text');
    }
    let from: number, to: number;
    if (start !== undefined || end !== undefined) {
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
        start! < 0 || end! > text.length || start! >= end! ||
        text.slice(start, end) !== word) {
        throw new Error('Privacy model supplied invalid or mismatched offsets');
      }
      from = start!; to = end!;
    } else {
      from = text.indexOf(word);
      if (from < 0) throw new Error('Privacy entity not located exactly in original text');
      if (text.indexOf(word, from + 1) >= 0) {
        throw new Error('Privacy entity is ambiguous in original text');
      }
      to = from + word.length;
    }
    // Decoded NER groups frequently contain the preceding whitespace. Preserve
    // separators in the original document; only sensitive characters are spans.
    const matched = text.slice(from,to);
    from += matched.length - matched.trimStart().length;
    to -= matched.length - matched.trimEnd().length;
    if (from >= to) throw new Error('Privacy model returned a whitespace-only entity');
    spans.push({category:entity_group, start:from, end:to, text:text.slice(from,to), confidence:score});
  }
  spans.sort((a,b) => a.start-b.start || a.end-b.end);
  for (let i=1;i<spans.length;i++) {
    if (spans[i].start < spans[i-1].end) throw new Error('Privacy entities overlap');
  }
  return spans;
}

/** Pure, optional masking. Never edits the input or stored source. */
export function redactPrivacySpans(
  text: string, spans: readonly PrivacySpan[], options: RedactOptions = {},
): string {
  if (typeof text !== 'string') throw new TypeError('Privacy input must be text');
  const ordered = [...spans].sort((a,b)=>a.start-b.start || a.end-b.end);
  let prevEnd = 0;
  for (const span of ordered) {
    if (!isPrivacyCategory(span.category) ||
      !Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.end) ||
      span.start < prevEnd || span.start < 0 || span.end > text.length ||
      span.end <= span.start || text.slice(span.start,span.end) !== span.text ||
      !Number.isFinite(span.confidence) || span.confidence < 0 || span.confidence > 1) {
      throw new Error('Invalid, overlapping, or stale privacy span');
    }
    prevEnd = span.end;
  }
  let out = '';
  let pos=0;
  for (const span of ordered) {
    if (options.categories && !options.categories.includes(span.category)) continue;
    out += text.slice(pos,span.start) + '[' + span.category.toUpperCase() + ']';
    pos=span.end;
  }
  return out + text.slice(pos);
}

interface EntityClassifier {
  (text: string, options: { aggregation_strategy: 'simple' }): Promise<unknown>;
  dispose(): Promise<void>;
}
interface TfModule {
  pipeline(task: string, model: string, options: {
    device: PrivacyDevice;
    dtype: PrivacyDtype;
    revision: string;
    progress_callback?: PrivacyFilterOptions['onProgress'];
  }): Promise<EntityClassifier>;
}

function abortError(): Error {
  const error = new Error('Privacy model load aborted');
  error.name = 'AbortError';
  return error;
}

export function createPrivacyFilterSubstrate(
  options: PrivacyFilterOptions = {},
): BrowserSubstrate<PrivacyInput, PrivacyResult> {
  const device: PrivacyDevice = options.device ?? 'webgpu';
  return {
    id:'openai/privacy-filter',
    format:'onnx',
    capabilities:['token-classification'],
    requestedBackends:[device],
    async create(_progress, signal) {
      if (signal?.aborted) throw abortError();
      const gpu = device === 'webgpu' && typeof navigator !== 'undefined'
        ? await navigator.gpu?.requestAdapter() : null;
      if (device === 'webgpu' && !gpu) throw new Error('Privacy model WebGPU adapter unavailable');
      const dtype = choosePrivacyFilterVariant(device, !!gpu?.features.has('shader-f16'), options.dtype);
      const transformers = await import('@huggingface/transformers') as unknown as TfModule;
      if (signal?.aborted) throw abortError();
      const classifier = await transformers.pipeline('token-classification', PRIVACY_MODEL_ID, {
        device,dtype,revision:PRIVACY_MODEL_REVISION,progress_callback:options.onProgress,
      });
      if (signal?.aborted) {
        await classifier.dispose();
        throw abortError();
      }
      return {
        async infer(input) {
          if (!input || typeof input.text !== 'string' || !input.text.trim()) {
            throw new Error('Privacy detection requires nonempty text');
          }
          const raw = await classifier(input.text,{aggregation_strategy:'simple'});
          return {kind:'privacy-detection',spans:resolvePrivacySpans(input.text,raw),device,dtype};
        },
        async dispose() { await classifier.dispose(); },
      };
    },
  };
}
