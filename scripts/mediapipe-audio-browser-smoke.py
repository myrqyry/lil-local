#!/usr/bin/env python3
"""End-to-end production-preview browser smoke for packed Lil Local YAMNet.

Uses official YAMNet model + real MediaPipe Audio WASM and a synthetic WAV.
Requires system Chromium and Python websocket-client. Never opens microphone.
"""
import json
import math
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent.parent
PORT = 4387
BASE = f'http://127.0.0.1:{PORT}'


def until(fn, title, seconds=50):
    end=time.monotonic()+seconds
    last=None
    while time.monotonic()<end:
        try:
            value=fn()
            if value:
                return value
        except Exception as exc:
            last=str(exc)
        time.sleep(.16)
    raise AssertionError('Timeout '+title+' '+str(last))


class Chrome:
    def __init__(self, ws):
        self.ws=ws
        self.seq=0
        self.errors=[]
        self.urls=[]

    def call(self, method, params=None):
        self.seq+=1
        mid=self.seq
        self.ws.send(json.dumps({'id':mid,'method':method,'params':params or {}}))
        while True:
            message=json.loads(self.ws.recv())
            name=message.get('method')
            body=message.get('params',{})
            if name=='Runtime.exceptionThrown':
                self.errors.append('exception '+str(body.get('exceptionDetails',{}))[:400])
            if name=='Runtime.consoleAPICalled' and body.get('type')=='error':
                self.errors.append('console '+str(body.get('args',[]))[:400])
            if name=='Network.responseReceived':
                self.urls.append(body.get('response',{}).get('url',''))
            if message.get('id')!=mid:
                continue
            if 'error' in message:
                raise RuntimeError(str(message['error']))
            return message.get('result',{})

    def eval(self, expression):
        data=self.call('Runtime.evaluate',{'expression':expression,'returnByValue':True})
        if data.get('exceptionDetails'):
            raise RuntimeError(str(data['exceptionDetails']))
        return data.get('result',{}).get('value')

    def button(self, text):
        js="""(() => {
          const button=[...document.querySelectorAll('button')].find(b=>
            b.getAttribute('aria-label')===LABEL || b.textContent.trim()===LABEL);
          if(!button || button.disabled) return false; button.click(); return true;
        })()""".replace('LABEL',json.dumps(text))
        return self.eval(js)

    def status(self):
        return self.eval("document.querySelector('[role=status]')?.textContent || ''")


def make_wav(target):
    with wave.open(str(target),'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        frames=[int(0.18*32767*math.sin(2*math.pi*440*i/16000))
                for i in range(16000*2)]
        handle.writeframes(struct.pack('<'+'h'*len(frames),*frames))


def main():
    work=Path(tempfile.mkdtemp(prefix='lil-local-audio-smoke-'))
    wav=work/'440-hz.wav'
    make_wav(wav)
    profile=work/'profile'
    preview=chrome=ws=None
    try:
        preview=subprocess.Popen(['pnpm','--filter','playground','exec','vite','preview',
            '--host','127.0.0.1','--port',str(PORT),'--strictPort'],
            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        until(lambda:urllib.request.urlopen(BASE,timeout=2).status==200,'preview',20)
        chrome=subprocess.Popen(['chromium','--headless=new','--no-sandbox',
            '--no-first-run','--no-default-browser-check','--disable-dev-shm-usage',
            '--no-proxy-server','--remote-debugging-port=0',
            '--remote-allow-origins=http://localhost',
            f'--user-data-dir={profile}','about:blank'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        port=until(lambda:(profile/'DevToolsActivePort').read_text().splitlines()[0]
            if (profile/'DevToolsActivePort').exists() else None,'chrome debugger',25)
        def page_ws():
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=2) as response:
                return next((row['webSocketDebuggerUrl'] for row in json.load(response)
                    if row.get('type')=='page'),None)
        ws=websocket.create_connection(until(page_ws,'page websocket',10),
            origin='http://localhost',timeout=90)
        c=Chrome(ws)
        for action in ['Page.enable','Runtime.enable','Network.enable','DOM.enable']:
            c.call(action)
        c.call('Page.navigate',{'url':BASE})
        label='Open YAMNet · MediaPipe Audio Classification'
        until(lambda:c.eval('!!document.querySelector(\'button[aria-label^="Open YAMNet"]\')'),'model card',30)
        assert not any('yamnet.tflite' in u for u in c.urls),'Model downloaded without Start'
        assert c.button(label)
        until(lambda:c.button('Load verified model'),'model panel ready',20)
        until(lambda:c.status()=='Ready · local audio classifier',
              'YAMNet official model + WASM loading',80)
        assert any('yamnet.tflite' in u for u in c.urls),'YAMNet weight request missing'
        assert any('audio_wasm_internal.wasm' in u for u in c.urls),'MediaPipe WASM missing'
        print('LOADED',c.status(),flush=True)

        dom=c.call('DOM.getDocument')
        file_node=c.call('DOM.querySelector',{
            'nodeId':dom['root']['nodeId'],
            'selector':'input[aria-label="Audio file for sound classification"]',
        })['nodeId']
        assert file_node,'Audio-file input missing'
        c.call('DOM.setFileInputFiles',{'nodeId':file_node,'files':[str(wav)]})
        c.eval("""(() => {
          const node=document.querySelector('input[aria-label="Audio file for sound classification"]');
          node.dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda:c.button('Classify locally'),'classify enabled',15)
        until(lambda:c.status().startswith('Classified '),'YAMNet inference',60)
        assert c.eval("document.querySelector('[aria-label=\"Sound classification results\"]')?.textContent.includes('Window')")
        assert not c.eval("!!document.querySelector('[role=alert]')"),'Error during inference'
        sample=c.eval("document.querySelector('[aria-label=\"Sound classification results\"]')?.textContent")
        print('OUTPUT',sample[:420],flush=True)
        assert c.button('Unload')
        until(lambda:c.status()=='Not loaded','dispose')
        failures=[item for item in c.errors if 'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.' not in item]
        assert not failures,'Browser errors: '+repr(failures)
        print('PASS: official SHA-verified model, real Chromium WASM/audio classification, unload',flush=True)
    finally:
        if ws: ws.close()
        for proc in [chrome,preview]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=4)
                except subprocess.TimeoutExpired:proc.kill()
        shutil.rmtree(work,ignore_errors=True)


if __name__=='__main__':
    main()
