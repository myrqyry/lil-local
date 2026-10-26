import { cp, mkdtemp, readdir, rm, mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const run = promisify(execFile)
const root = resolve(new URL('..', import.meta.url).pathname)
const scratch = await mkdtemp(join(tmpdir(), 'lil-local-sharp-consumer-'))
try {
  const archives = join(scratch, 'archives')
  const src = join(scratch, 'src')
  await mkdir(archives, { recursive: true })
  await mkdir(src)
  for (const pkg of ['inference-core', 'sharp-onnx']) {
    const staging = join(scratch, 'pack-' + pkg)
    await mkdir(staging)
    await run('pnpm', ['pack', '--pack-destination', staging], {cwd: join(root, 'packages', pkg)})
    const [file] = await readdir(staging)
    await cp(join(staging, file), join(archives, pkg + '.tgz'))
  }
  await writeFile(join(scratch, 'package.json'), JSON.stringify({
    name: 'sharp-packed-consumer-test', private: true, type: 'module',
    scripts: { typecheck: 'tsc --noEmit', build: 'vite build' },
    dependencies: {
      '@lil-local/inference-core': 'file:./archives/inference-core.tgz',
      '@lil-local/sharp-onnx': 'file:./archives/sharp-onnx.tgz',
    },
    devDependencies: { typescript: '^7.0.2', vite: '^8.1.5' },
  }, null, 2))
  await writeFile(join(scratch, 'tsconfig.json'), JSON.stringify({
    compilerOptions: {target:'ES2020',module:'ESNext',moduleResolution:'bundler',strict:true,
      lib:['ES2020','DOM','DOM.Iterable'],types:[],skipLibCheck:true,noEmit:true},
    include:['src'],
  }))
  await writeFile(join(scratch, 'index.html'), '<div id="app"></div><script type="module" src="/src/index.ts"></script>')
  await writeFile(join(src, 'index.ts'), `
    import { BrowserSubstrateRunner } from '@lil-local/inference-core'
    import { createSharpSubstrate, buildSharpPly, type SharpRunInput } from '@lil-local/sharp-onnx'
    const pipeline = new BrowserSubstrateRunner(createSharpSubstrate({
      graphUrl:'/models/sharp_web_predictor.onnx',
      externalDataUrl:'/models/sharp_web_predictor.onnx.data',
    }))
    export async function generate(image: SharpRunInput): Promise<Uint8Array> {
      await pipeline.load()
      const output = await pipeline.run(image)
      await pipeline.dispose()
      return buildSharpPly(output.gaussians)
    }
  `)
  await run('pnpm', ['install', '--ignore-workspace', '--no-frozen-lockfile', '--ignore-scripts', '--offline'], {cwd:scratch})
  await run('pnpm', ['run','typecheck'], {cwd:scratch})
  await run('pnpm', ['run','build'], {cwd:scratch})
  console.log('Sharp ONNX packed consumer: install, TS typecheck, Vite build PASSED')
} finally {
  await rm(scratch, {recursive:true, force:true})
}
