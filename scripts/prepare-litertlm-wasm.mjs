#!/usr/bin/env node
/** Self-host the exact pinned LiteRT-LM 0.19.0 JS/WASM runtime in the playground.
 * Generated vendor assets are gitignored, never manually edited or fetched by
 * the application at inference time. Source remains the installed pnpm package.
 */
import {createHash} from 'node:crypto'
import {copyFile, mkdir, readFile, stat} from 'node:fs/promises'
import {realpathSync} from 'node:fs'
import {join} from 'node:path'
import {fileURLToPath} from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const sdk = realpathSync(join(root, 'packages/text-gen/node_modules/@litert-lm/core'))
const pkg = JSON.parse(await readFile(join(sdk, 'package.json'), 'utf8'))
if (pkg.version !== '0.19.0') throw new Error('Expected pinned @litert-lm/core 0.19.0, found ' + pkg.version)

const source = join(sdk, 'wasm')
const destination = join(root, 'apps/playground/public/vendor/litert-lm-0.19.0')
const variants = [
  'litertlm_wasm_internal',
  'litertlm_wasm_asyncify_internal',
  'litertlm_wasm_compat_internal',
  'litertlm_wasm_compat_asyncify_internal',
]
const names = variants.flatMap(variant => [variant + '.js', variant + '.wasm'])
await mkdir(destination, {recursive: true})
let copied = 0
for (const name of names) {
  const src = join(source, name)
  const dst = join(destination, name)
  const from = await readFile(src)
  const expected = createHash('sha256').update(from).digest('hex')
  let actual
  try {
    if ((await stat(dst)).size === from.byteLength) {
      actual = createHash('sha256').update(await readFile(dst)).digest('hex')
    }
  } catch { /* missing destination: copy */ }
  if (actual !== expected) {
    await copyFile(src, dst)
    const copiedHash = createHash('sha256').update(await readFile(dst)).digest('hex')
    if (copiedHash !== expected) throw new Error('LiteRT-LM runtime asset mismatch: ' + name)
    copied++
  }
}
console.log('Local LiteRT-LM 0.19.0 WASM ready: ' + names.length + ' verified assets; ' + copied + ' copied')
