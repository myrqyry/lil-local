#!/usr/bin/env node
/** Self-host the ORT pair from the exact Transformers.js dependency used by Lil Local depth. */
import {createRequire} from 'node:module'
import {createHash} from 'node:crypto'
import {mkdir,copyFile,readFile} from 'node:fs/promises'
import {dirname,resolve,join} from 'node:path'
import {fileURLToPath} from 'node:url'

const root=fileURLToPath(new URL('../',import.meta.url))
const require=createRequire(resolve(root,'packages/depth-estimation/package.json'))
const transformersEntry=require.resolve('@huggingface/transformers')
const viaTransformers=createRequire(transformersEntry)
const distribution=dirname(viaTransformers.resolve('onnxruntime-web/webgpu'))
const destination=join(root,'apps/playground/public/ort')
await mkdir(destination,{recursive:true})
for(const name of ['ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm']){
  const original=join(distribution,name)
  const local=join(destination,name)
  await copyFile(original,local)
  const src=createHash('sha256').update(await readFile(original)).digest('hex')
  const dst=createHash('sha256').update(await readFile(local)).digest('hex')
  if(src!==dst)throw new Error('ONNX runtime asset copy verification failed: '+name)
  console.log('Verified local ONNX runtime',name,(await readFile(local)).byteLength,'bytes')
}
