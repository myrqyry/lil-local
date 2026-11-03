import { cp, mkdtemp, readdir, rm, mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
const run = promisify(execFile)
const root = resolve(new URL('..', import.meta.url).pathname)
const temp = await mkdtemp(join(tmpdir(), 'lil-local-privacy-consumer-'))
try {
  const archive = join(temp, 'archives')
  await mkdir(archive)
  await mkdir(join(temp, 'src'))
  for (const name of ['inference-core', 'privacy-filter']) {
    const dir = join(temp, 'pack-' + name)
    await mkdir(dir)
    await run('pnpm', ['pack', '--pack-destination', dir], { cwd: join(root, 'packages', name) })
    const [file] = await readdir(dir)
    await cp(join(dir, file), join(archive, name + '.tgz'))
  }
  await writeFile(join(temp, 'package.json'), JSON.stringify({
    name: 'privacy-filter-external-consumer', private: true, type: 'module',
    scripts: { typecheck: 'tsc --noEmit', build: 'vite build' },
    dependencies: {
      '@lil-local/inference-core': 'file:./archives/inference-core.tgz',
      '@lil-local/privacy-filter': 'file:./archives/privacy-filter.tgz',
    },
    devDependencies: { typescript: '^7.0.2', vite: '^8.1.5' },
  }, null, 2))
  await writeFile(join(temp, 'tsconfig.json'), JSON.stringify({
    compilerOptions: { target: 'ES2020', module:'ESNext', moduleResolution:'bundler',
    strict:true, lib:['ES2020','DOM','DOM.Iterable'], skipLibCheck:true, noEmit:true },
    include:['src'],
  }))
  await writeFile(join(temp, 'index.html'), '<script type="module" src="/src/main.ts"></script>')
  await writeFile(join(temp, 'src/main.ts'), [
    "import { BrowserSubstrateRunner } from '@lil-local/inference-core'",
    "import { createPrivacyFilterSubstrate, redactPrivacySpans, type PrivacyResult } from '@lil-local/privacy-filter'",
    "export const runner = new BrowserSubstrateRunner(createPrivacyFilterSubstrate({device:'webgpu'}))",
    "export function mask(text: string, result: PrivacyResult): string {",
    "  return redactPrivacySpans(text, result.spans)",
    "}",
  ].join('\n'))
  await run('pnpm', ['install','--ignore-workspace','--no-frozen-lockfile','--ignore-scripts','--offline'],{cwd:temp})
  await run('pnpm', ['run','typecheck'],{cwd:temp})
  await run('pnpm', ['run','build'],{cwd:temp})
  console.log('Privacy Filter packed consumer: install, types, Vite build PASS')
} finally { await rm(temp,{recursive:true,force:true}) }
