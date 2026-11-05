#!/usr/bin/env python3
"""Real Chromium qualification for Lil Local MediaPipe universal retrieval.

Requires the official EmbeddingGemma 2 text-only model at LIL_LOCAL_RETRIEVAL_MODEL
(default: ~/.cache/lil-local/mediapipe-retrieval/embeddinggemma-2-text-270m.litertlm).
No weight download is implicit, model bytes are checked against a pinned SHA-256,
and the browser runs from a disposable profile. Python websocket-client required.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
import websocket

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get(
    'LIL_LOCAL_RETRIEVAL_MODEL',
    str(Path.home() / '.cache/lil-local/mediapipe-retrieval/embeddinggemma-2-text-270m.litertlm'),
))
MODEL_BYTES=164_626_432
MODEL_SHA='2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb'
BASE='http://127.0.0.1:4392'

def until(fn, title, duration=30):
    end=time.monotonic()+duration
    last=None
    while time.monotonic()<end:
        try:
            value=fn()
            if value:
                return value
        except Exception as exc:
            last=str(exc)
        time.sleep(.25)
    raise RuntimeError(f'Timed out: {title}, last={last}')

class Browser:
    def __init__(self,ws):
        self.ws=ws
        self.index=0
        self.errors=[]
        self.urls=[]

    def call(self,method,params=None):
        self.index+=1
        ident=self.index
        self.ws.send(json.dumps({'id':ident,'method':method,'params':params or {}}))
        while True:
            item=json.loads(self.ws.recv())
            name=item.get('method','')
            payload=item.get('params',{})
            if name=='Runtime.exceptionThrown':
                self.errors.append('exception '+str(payload.get('exceptionDetails',{}))[:400])
            if name=='Runtime.consoleAPICalled' and payload.get('type')=='error':
                self.errors.append('console '+str(payload.get('args',[]))[:400])
            if name=='Network.responseReceived':
                self.urls.append(payload.get('response',{}).get('url',''))
            if item.get('id')!=ident:
                continue
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            return item.get('result',{})

    def eval(self,script):
        response=self.call('Runtime.evaluate',{
            'expression':script,'returnByValue':True,'awaitPromise':True})
        if response.get('exceptionDetails'):
            raise RuntimeError(str(response['exceptionDetails']))
        return response.get('result',{}).get('value')

    def button(self,label):
        return self.eval("""(() => {
          const button=[...document.querySelectorAll('button')].find(x=>
            x.getAttribute('aria-label')===LABEL || x.textContent.trim()===LABEL);
          if(!button || button.disabled)return false;
          button.click();return true;
        })()""".replace('LABEL',json.dumps(label)))

    def status(self):
        return self.eval("document.querySelector('[role=status]')?.textContent || ''")

    def error(self):
        return self.eval("document.querySelector('[role=alert]')?.textContent || ''")

def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES:
        raise RuntimeError(f'Missing exact official 164626432-byte model file: {MODEL}')
    digest=sha256_file(MODEL)
    if digest!=MODEL_SHA:
        raise RuntimeError(f'Model SHA-256 mismatch: {digest}')
    print('CHECKPOINT_SHA',digest,flush=True)
    directory=Path(tempfile.mkdtemp(prefix='lil-local-mediapipe-retrieval-smoke-'))
    profile=directory/'chrome'
    preview=chrome=connection=None
    try:
        preview=subprocess.Popen([
          'pnpm','--filter','playground','exec','vite','preview',
          '--host','127.0.0.1','--port','4392','--strictPort'],
          cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        until(lambda:urllib.request.urlopen(BASE,timeout=2).status==200,'preview',25)
        chrome=subprocess.Popen([
          'chromium','--headless=new','--no-sandbox','--disable-dev-shm-usage',
          '--no-proxy-server','--no-first-run','--no-default-browser-check',
          '--enable-unsafe-webgpu','--remote-debugging-port=0',
          '--remote-allow-origins=http://localhost',f'--user-data-dir={profile}',
          'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        port=until(lambda:(profile/'DevToolsActivePort').read_text().splitlines()[0]
            if (profile/'DevToolsActivePort').exists() else None,'Chrome DevTools',25)
        def endpoint():
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=2) as response:
                return next((x['webSocketDebuggerUrl'] for x in json.load(response)
                    if x.get('type')=='page'),None)
        connection=websocket.create_connection(until(endpoint,'DevTools page',15),
            origin='http://localhost',timeout=210)
        c=Browser(connection)
        for name in ['Page.enable','Runtime.enable','Network.enable','DOM.enable']:
            c.call(name)
        c.call('Page.navigate',{'url':BASE})
        label='Open Universal Embedder · MediaPipe Search'
        until(lambda:c.eval("""[...document.querySelectorAll('button')]
          .some(b=>b.getAttribute('aria-label')==='LABEL')""".replace('LABEL',label)),
          'Universal Retriever model list entry',40)
        assert not any('embeddinggemma-2-text-270m.litertlm' in u for u in c.urls),'Unexpected model auto-download'
        assert c.button(label)
        until(lambda:c.eval('!!document.querySelector(\'input[aria-label="LiteRT-LM retrieval model file"]\')'),
              'reusable consumer panel',25)
        gpu=c.eval("(async () => {const a=await navigator.gpu?.requestAdapter(); return {supported:!!navigator.gpu,adapter:!!a,device:a?.info?.description??a?.info?.vendor??'unknown'}})()")
        print('WEBGPU',gpu,flush=True)

        document=c.call('DOM.getDocument')
        node=c.call('DOM.querySelector',{
          'nodeId':document['root']['nodeId'],
          'selector':'input[aria-label="LiteRT-LM retrieval model file"]',
        })['nodeId']
        if not node: raise RuntimeError('Model-file input not found')
        c.call('DOM.setFileInputFiles',{'nodeId':node,'files':[str(MODEL)]})
        c.eval("""(() => {
          document.querySelector('input[aria-label="LiteRT-LM retrieval model file"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda:c.button('Load local retrieval'),'Load opt-in',25)
        print('STARTED',c.status(),flush=True)
        try:
            until(lambda:c.status().startswith('Ready · embeddings') or bool(c.error()),
                  'native retrieval load',165)
        except Exception:
            print('LOAD_STATE',c.status(),'ERROR',c.error(),flush=True)
            raise
        if c.error():
            raise RuntimeError('MediaPipe native model initialization failed: '+c.error())
        print('READY',c.status(),flush=True)

        assert c.button('Embed query')
        until(lambda:c.status()=='Local embed-text complete' or bool(c.error()),
              'real universal embedding',70)
        if c.error():raise RuntimeError('Embedding failed: '+c.error())
        text=c.eval("document.querySelector('[aria-label=\"MediaPipe retrieval output\"]')?.textContent||''")
        print('EMBED_RESULT',text[:300],flush=True)
        assert 'dimensions' in text and 'Embedding 1' in text
        assert c.button('Index document')
        until(lambda:c.status()=='Local insert-text complete' or bool(c.error()),
              'real semantic index',85)
        if c.error():raise RuntimeError('Index failed: '+c.error())
        assert c.button('Search locally')
        until(lambda:c.status()=='Local search complete' or bool(c.error()),
              'real semantic search',80)
        if c.error():raise RuntimeError('Search failed: '+c.error())
        matches=c.eval("document.querySelector('[aria-label=\"MediaPipe retrieval output\"]')?.textContent||''")
        print('SEARCH_RESULT',matches[:320],flush=True)
        assert 'doc-1' in matches
        assert c.button('Unload & clear in-memory index')
        until(lambda:c.status()=='Not loaded','dispose',30)
        failures=[e for e in c.errors
                  if 'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.' not in e]
        print('BROWSER_ERRORS',failures[-5:],flush=True)
        assert not failures,'Browser runtime errors: '+repr(failures)
        print('PASS: pinned model, MediaPipe embeddings with WebGPU available, native index/search, unload',flush=True)
    finally:
        if connection:connection.close()
        for process in [chrome,preview]:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill()
        shutil.rmtree(directory,ignore_errors=True)

if __name__=='__main__':
    main()
