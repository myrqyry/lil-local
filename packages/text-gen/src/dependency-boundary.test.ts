import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

async function sourceFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true })
  const files: string[] = []
  for (const entry of entries) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) files.push(...await sourceFiles(path))
    else if (/\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith('dependency-boundary.test.ts')) files.push(path)
  }
  return files
}

describe('text-gen dependency boundary', () => {
  it('does not import runtime or model-specific code', async () => {
    const files = await sourceFiles(new URL('.', import.meta.url).pathname)
    const contents = await Promise.all(files.map((file) => readFile(file, 'utf8')))
    const forbidden = /(?:@litertjs\/core|@lil-local\/(?:runtime-litert|kokoro|qwen3-tts)|apps\/playground|podqast)/
    expect(contents.some((content) => forbidden.test(content))).toBe(false)
  })
})
