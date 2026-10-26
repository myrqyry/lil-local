import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BrowserSubstrateRunner } from '@lil-local/inference-core';
import { choosePrivacyFilterVariant, resolvePrivacySpans, redactPrivacySpans, createPrivacyFilterSubstrate, PRIVACY_MODEL_REVISION } from './index';

const mocked = vi.hoisted(() => ({ fromPipeline: vi.fn(), load: vi.fn(), run: vi.fn(), dispose: vi.fn() }));
vi.mock('@huggingface/transformers', () => ({ pipeline: mocked.fromPipeline }));

describe('Privacy Filter detection and masking contract', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mocked.run.mockResolvedValue([{ entity_group:'private_email', score:.99, word:' maya@example.com' }]);
    mocked.fromPipeline.mockImplementation(async () => {
      mocked.load();
      const classifier=async (input: string, options: unknown) => mocked.run(input, options);
      classifier.dispose=mocked.dispose;
      return classifier;
    });
  });
  it('selects smallest compatible model, retaining caller options', () => {
    expect(choosePrivacyFilterVariant('webgpu',true)).toBe('q4f16');
    expect(choosePrivacyFilterVariant('webgpu',false)).toBe('q4');
    expect(choosePrivacyFilterVariant('wasm',false)).toBe('q4');
    expect(choosePrivacyFilterVariant('webgpu',true,'fp16')).toBe('fp16');
    expect(()=>choosePrivacyFilterVariant('webgpu',false,'q4f16')).toThrow(/shader-f16/);
  });
  it('maps only uniquely matched exact spans into the original input', () => {
    const text='Email maya@example.com and continue.';
    const spans=resolvePrivacySpans(text,[{entity_group:'private_email',score:.99,word:' maya@example.com'}]);
    expect(spans).toEqual([{category:'private_email',start:6,end:22,confidence:.99,text:'maya@example.com'}]);
    expect(redactPrivacySpans(text,spans)).toBe('Email [PRIVATE_EMAIL] and continue.');
    expect(text).toBe('Email maya@example.com and continue.');
  });
  it('rejects ambiguous repeated values rather than masking an arbitrary occurrence', () => {
    expect(()=>resolvePrivacySpans('Alice and Alice',[{entity_group:'private_person',score:.9,word:'Alice'}])).toThrow(/ambiguous/);
  });
  it('rejects malformed, missing, overlapping, or out-of-range data', () => {
    expect(()=>resolvePrivacySpans('hi',[{entity_group:'private_person',score:NaN,word:'hi'}])).toThrow();
    expect(()=>resolvePrivacySpans('hi',[{entity_group:'private_person',score:.9,word:'Bob'}])).toThrow(/not located/);
    expect(()=>redactPrivacySpans('a',[{category:'private_person',start:0,end:2,text:'a',confidence:.9}])).toThrow();
  });
  it('supports selection by categories without editing source', () => {
    const text='mail me@a.io 555-1111';
    const spans=resolvePrivacySpans(text,[{entity_group:'private_email',score:.9,word:'me@a.io'},{entity_group:'private_phone',score:.8,word:'555-1111'}]);
    expect(redactPrivacySpans(text,spans,{categories:['private_phone']})).toBe('mail me@a.io [PRIVATE_PHONE]');
  });
  it('never loads the model until the consuming runner explicitly loads', async () => {
    const runner=new BrowserSubstrateRunner(createPrivacyFilterSubstrate({device:'wasm',dtype:'q4'}));
    expect(runner.status).toBe('idle');
    expect(mocked.fromPipeline).not.toHaveBeenCalled();
    await runner.load();
    expect(mocked.fromPipeline).toHaveBeenCalledWith('token-classification','openai/privacy-filter',expect.objectContaining({device:'wasm',dtype:'q4',revision:PRIVACY_MODEL_REVISION}));
    const result=await runner.run({text:'Email maya@example.com and continue.'});
    expect(result.kind).toBe('privacy-detection');
    expect(result.spans[0].category).toBe('private_email');
    expect(mocked.run).toHaveBeenCalledWith('Email maya@example.com and continue.',{aggregation_strategy:'simple'});
    await runner.dispose();
    expect(mocked.dispose).toHaveBeenCalledTimes(1);
  });
  it('does not swallow offset-integrity errors as empty privacy results', async () => {
    mocked.run.mockResolvedValue([{entity_group:'private_person',score:.9,word:'missing'}]);
    const runner=new BrowserSubstrateRunner(createPrivacyFilterSubstrate({device:'wasm'}));
    await runner.load();
    await expect(runner.run({text:'hello'})).rejects.toThrow(/not located/);
    await runner.dispose();
  });
});
