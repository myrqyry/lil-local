#!/usr/bin/env python3
"""Real Chromium MediaPipe language detection, with explicit native model load.

Disposable profile; no user camera or uploaded text. Runs the actual 315 KB
publisher checkpoint and matching Tasks Text WASM on English/Spanish/Japanese.
"""
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

ROOT=Path(__file__).resolve().parent.parent
BASE='http://127.0.0.1:4397'
MODEL='language_detector.tflite'

def until(fn,label,seconds=50):
    deadline=time.monotonic()+seconds
    last=None
    while time.monotonic()<deadline:
        try:
            value=fn()
            if value:return value
        except Exception as e:last=repr(e)
        time.sleep(.2)
    raise RuntimeError(f'Timeout waiting for {label}: {last}')

class Browser:
    def __init__(self,conn):
        self.ws=conn
        self.next=0
        self.urls=[]
        self.errors=[]
    def send(self,method,params=None):
        self.next+=1
        ident=self.next
        self.ws.send(json.dumps({'id':ident,'method':method,'params':params or {}}))
        while True:
            value=json.loads(self.ws.recv())
            event=value.get('method','')
            payload=value.get('params',{})
            if event=='Network.responseReceived':
                self.urls.append(payload.get('response',{}).get('url',''))
            if event=='Runtime.exceptionThrown':
                self.errors.append('exception '+str(payload.get('exceptionDetails',{}))[:360])
            if event=='Runtime.consoleAPICalled' and payload.get('type')=='error':
                self.errors.append('console '+str(payload.get('args',[]))[:360])
            if value.get('id')!=ident:continue
            if 'error' in value:raise RuntimeError(str(value['error']))
            return value.get('result',{})
    def eval(self,js):
        response=self.send('Runtime.evaluate',{
            'expression':js,'returnByValue':True,'awaitPromise':True})
        if response.get('exceptionDetails'):
            raise RuntimeError(str(response['exceptionDetails']))
        return response.get('result',{}).get('value')
    def click(self,label):
        return self.eval("""(() => {
          const button=[...document.querySelectorAll('button')].find(b=>
            b.getAttribute('aria-label')===LABEL || b.textContent.trim()===LABEL);
          if(!button||button.disabled)return false;
          button.click();return true;
        })()""".replace('LABEL',json.dumps(label)))
    def status(self):
        return self.eval("document.querySelector('[role=status]')?.textContent||''")
    def error(self):
        return self.eval("document.querySelector('[role=alert]')?.textContent||''")
    def detected(self):
        return self.eval("""(() => {
          const root=document.querySelector('[aria-label="Language prediction results"]');
          const top=root?.querySelector('strong')?.textContent||'';
          return {top,detail:root?.textContent||''};
        })()""")

def main():
    directory=Path(tempfile.mkdtemp(prefix='lil-local-language-real-'))
    server=chrome=connection=None
    try:
        server=subprocess.Popen([
            'pnpm','--filter','playground','exec','vite','preview',
            '--host','127.0.0.1','--port','4397','--strictPort',
        ],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        until(lambda:urllib.request.urlopen(BASE,timeout=2).status==200,
              'production preview',30)
        profile=directory/'profile'
        chrome=subprocess.Popen([
            'chromium','--headless=new','--no-sandbox','--disable-dev-shm-usage',
            '--no-proxy-server','--no-first-run','--no-default-browser-check',
            '--enable-unsafe-swiftshader','--use-gl=angle',
            '--use-angle=swiftshader-webgl',
            '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
            '--remote-allow-origins=http://localhost',f'--user-data-dir={profile}',
            'about:blank',
        ],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        port=until(lambda:(profile/'DevToolsActivePort').read_text().splitlines()[0]
            if (profile/'DevToolsActivePort').exists() else None,'debugger port',30)
        def ws_url():
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=3) as response:
                return next((entry['webSocketDebuggerUrl'] for entry in json.load(response)
                    if entry.get('type')=='page'),None)
        connection=websocket.create_connection(until(ws_url,'page websocket',15),
            origin='http://localhost',timeout=110)
        browser=Browser(connection)
        for method in ('Page.enable','Runtime.enable','Network.enable'):
            browser.send(method)
        browser.send('Page.navigate',{'url':BASE})
        card='Open Language Detector · MediaPipe Text'
        until(lambda:browser.eval("""[...document.querySelectorAll('button')].some(
          b=>b.getAttribute('aria-label')===LABEL)""".replace('LABEL',json.dumps(card))),
          'language model card',35)
        assert not any(MODEL in url for url in browser.urls), (
            'Language detector weights fetched before explicit action')
        assert browser.click(card)
        until(lambda:browser.eval(
            '!!document.querySelector(\'textarea[aria-label="Text for local language identification"]\')'),
            'language workbench',15)
        assert browser.status().startswith('Model not loaded')
        assert not any(MODEL in url for url in browser.urls), (
            'Language model fetched on opening the workbench')
        assert browser.click('Load language detector')
        until(lambda:browser.status().startswith('Local language detector ready') or browser.error(),
              'native MediaPipe Tasks Text GPU/WASM init',100)
        if browser.error():raise RuntimeError('Real model failed: '+browser.error())
        assert any(MODEL in url for url in browser.urls), (
            'Publisher language model was never fetched')
        assert any('text_wasm_internal.wasm' in url for url in browser.urls), (
            'Matching Tasks Text WASM was not fetched')
        print('REAL_LOAD',browser.status(),flush=True)
        for label,expected in [('English','en'),('Español','es'),('日本語','ja')]:
            assert browser.click(label)
            assert browser.click('Detect language')
            until(lambda:browser.status().startswith('Text classified locally') or browser.error(),
                  label+' real prediction',35)
            if browser.error():raise RuntimeError(label+' inference: '+browser.error())
            value=browser.detected()
            print('REAL_LANGUAGE',label,value,flush=True)
            assert value['top']==expected,(label,'expected',expected,'observed',value)
        assert browser.click('Unload')
        until(lambda:browser.status()=='Model unloaded.','native unload',25)
        problems=[item for item in browser.errors if not any(marker in item for marker in (
            'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.',
            'gl_context.cc:408] GL version:',
            'gl_context.cc:1136] OpenGL error checking is disabled',
            'inference_feedback_manager.cc:121] Feedback manager requires',
            'gl_context_webgl.cc:149] Successfully destroyed WebGL context',
        ))]
        print('BROWSER_ERRORS',problems,flush=True)
        assert not problems, 'Unexpected browser errors'
        print('PASS: actual 315 KB model, verified English/Spanish/Japanese, cold load, native WebGL delegate, unload',flush=True)
    finally:
        if connection:connection.close()
        for process in (chrome,server):
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=4)
                except subprocess.TimeoutExpired:process.kill()
        shutil.rmtree(directory,ignore_errors=True)

if __name__=='__main__':
    main()
