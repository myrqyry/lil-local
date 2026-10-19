import { readFile, readdir } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';

async function sourceFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const full = `${directory}/${entry.name}`;
    if (entry.isDirectory()) {
      files.push(...(await sourceFiles(full)));
    } else if (/\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith('dependency-boundary.test.ts')) {
      files.push(full);
    }
  }
  return files;
}

const FORBIDDEN = /@lil-local\/(?:kokoro|qwen3-tts|text-gen|retrieval|encoder)|apps\/playground|podqast/;

describe('video-classification dependency boundary', () => {
  it('does not import runtime or model-specific code', async () => {
    const files = await sourceFiles(new URL('.', import.meta.url).pathname);
    const contents = await Promise.all(files.map(file => readFile(file, 'utf8')));
    expect(contents.some(content => FORBIDDEN.test(content))).toBe(false);
  });
});
