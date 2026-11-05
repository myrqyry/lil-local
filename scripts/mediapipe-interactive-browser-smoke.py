#!/usr/bin/env python3
"""Real MagicTouch V2 Chromium test: one image encoding, two stroke refinements.

Runs in disposable Chromium with real publisher model and SDK WASM; it never
requests camera permissions. Model weights and the upstream sample photo are
not written into the repo. Needs chromium and Python websocket-client.
"""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
import websocket

ROOT=Path(__file__).resolve().parent.parent
BASE='http://127.0.0.1:4394'
SOURCE='6887f6399745dc549dadd0022c31fc9ab01f26d6'
MODEL_NAME='interactive_segmentation.task'
MODEL_SHA='38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03'
DOG_SHA='87295aa79a9e285f86a94c58230c50401453dbc8cca946d544ce0e3de3f71cf7'

def until(check,label,seconds=50):
    deadline=time.monotonic()+seconds
    last=None
    while time.monotonic()<deadline:
        try:
            value=check()
            if value:return value
        except Exception as e:last=str(e)
        time.sleep(.2)
    raise RuntimeError(f'Timed out waiting for {label}: {last}')

class Browser:
    def __init__(self,connection):
        self.ws=connection
        self.seq=0
        self.errors=[]
        self.urls=[]

    def send(self,method,params=None):
        self.seq+=1
        expected=self.seq
        self.ws.send(json.dumps({'id':expected,'method':method,'params':params or {}}))
        while True:
            record=json.loads(self.ws.recv())
            name=record.get('method','')
            payload=record.get('params',{})
            if name=='Runtime.exceptionThrown':
                self.errors.append('exception '+str(payload.get('exceptionDetails',{}))[:360])
            if name=='Runtime.consoleAPICalled' and payload.get('type')=='error':
                self.errors.append('console '+str(payload.get('args',[]))[:360])
            if name=='Network.responseReceived':
                self.urls.append(payload.get('response',{}).get('url',''))
            if record.get('id')!=expected:continue
            if 'error' in record:raise RuntimeError(str(record['error']))
            return record.get('result',{})

    def eval(self,script):
        result=self.send('Runtime.evaluate',{
          'expression':script,'returnByValue':True,'awaitPromise':True,
        })
        if result.get('exceptionDetails'):
            raise RuntimeError(str(result['exceptionDetails']))
        return result.get('result',{}).get('value')

    def click(self,label):
        return self.eval("""(() => {
          const button=[...document.querySelectorAll('button')].find(x=>
            x.getAttribute('aria-label')===LABEL || x.textContent.trim()===LABEL);
          if (!button||button.disabled)return false;
          button.click();return true;
        })()""".replace('LABEL',json.dumps(label)))

    def status(self):
        return self.eval("document.querySelector('[role=status]')?.textContent||''")

    def error(self):
        return self.eval("document.querySelector('[role=alert]')?.textContent||''")

    def drag(self,start,end,steps=12):
        selector='svg[aria-label="Draw MagicTouch strokes over image"]'
        rect=self.eval("""(() => {
          const svg=document.querySelector(SELECTOR);
          svg?.scrollIntoView({block:'center'});
          const r=svg?.getBoundingClientRect();
          return r?{x:r.x,y:r.y,width:r.width,height:r.height}:null;
        })()""".replace('SELECTOR',json.dumps(selector)))
        if not rect:raise RuntimeError('Interactive drawing canvas is missing')
        def pos(p):
            return (rect['x']+p[0]*rect['width'],
                    rect['y']+p[1]*rect['height'])
        x,y=pos(start)
        self.send('Input.dispatchMouseEvent',{'type':'mouseMoved','x':x,'y':y})
        self.send('Input.dispatchMouseEvent',{'type':'mousePressed','x':x,'y':y,
               'button':'left','clickCount':1})
        for i in range(1,steps+1):
            t=i/steps
            xx,yy=pos((start[0]*(1-t)+end[0]*t,start[1]*(1-t)+end[1]*t))
            self.send('Input.dispatchMouseEvent',{
              'type':'mouseMoved','x':xx,'y':yy,'button':'left','buttons':1})
        self.send('Input.dispatchMouseEvent',{
          'type':'mouseReleased','x':xx,'y':yy,'button':'left','clickCount':1})

def main():
    folder=Path(tempfile.mkdtemp(prefix='lil-local-magic-touch-real-smoke-'))
    preview=chrome=connection=None
    try:
        dog=folder/'dog.jpg'
        url=f'https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples-web/{SOURCE}/public/dog.jpg'
        with urllib.request.urlopen(url,timeout=25) as response:
            dog.write_bytes(response.read())
        assert hashlib.sha256(dog.read_bytes()).hexdigest()==DOG_SHA
        preview=subprocess.Popen([
            'pnpm','--filter','playground','exec','vite','preview',
            '--host','127.0.0.1','--port','4394','--strictPort',
        ],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        until(lambda:urllib.request.urlopen(BASE,timeout=2).status==200,'preview',30)
        profile=folder/'profile'
        chrome=subprocess.Popen([
            'chromium','--headless=new','--no-sandbox','--disable-dev-shm-usage',
            '--no-proxy-server','--no-first-run','--no-default-browser-check',
            '--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader-webgl',
            '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
            '--remote-allow-origins=http://localhost',f'--user-data-dir={profile}',
            'about:blank',
        ],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        port=until(lambda:(profile/'DevToolsActivePort').read_text().splitlines()[0]
            if (profile/'DevToolsActivePort').exists() else None,'Chrome debugger',30)
        def endpoint():
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=2) as response:
                return next((entry['webSocketDebuggerUrl'] for entry in json.load(response)
                    if entry.get('type')=='page'),None)
        connection=websocket.create_connection(until(endpoint,'Chrome page',15),
            origin='http://localhost',timeout=190)
        browser=Browser(connection)
        for method in ['Page.enable','Runtime.enable','Network.enable','DOM.enable']:
            browser.send(method)
        browser.send('Emulation.setDeviceMetricsOverride',{
            'width':1100,'height':1400,'deviceScaleFactor':1,'mobile':False})
        browser.send('Page.navigate',{'url':BASE})
        entry='Open MagicTouch V2 · Interactive Segmentation'
        until(lambda:browser.eval("""[...document.querySelectorAll('button')].some(
            b=>b.getAttribute('aria-label')===LABEL)""".replace('LABEL',json.dumps(entry))),
              'MagicTouch entry',35)
        assert not any(MODEL_NAME in u for u in browser.urls), 'Model downloaded during discovery'
        assert browser.click(entry)
        until(lambda:browser.eval("!!document.querySelector('input[aria-label=\"MagicTouch input image\"]')"),
              'MagicTouch consumer',15)
        dom=browser.send('DOM.getDocument')
        node=browser.send('DOM.querySelector',{
            'nodeId':dom['root']['nodeId'],
            'selector':'input[aria-label="MagicTouch input image"]',
        })['nodeId']
        assert node
        browser.send('DOM.setFileInputFiles',{'nodeId':node,'files':[str(dog)]})
        browser.eval("""(() => {
          document.querySelector('input[aria-label="MagicTouch input image"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda:browser.click('Load & encode image'),'opt-in load button',15)
        start=time.monotonic()
        try:
            until(lambda:browser.status().startswith('Image encoded.') or browser.error(),
                  'real MagicTouch image encoder',150)
        except Exception:
            print('LOAD_ERROR',browser.status(),browser.error(),flush=True)
            raise
        if browser.error():raise RuntimeError('Native load failed: '+browser.error())
        print('ENCODE_COMPLETE_S',round(time.monotonic()-start,2),browser.status(),flush=True)
        assert any(MODEL_NAME in u for u in browser.urls),'No official model fetch observed'
        first=time.monotonic()
        browser.drag((.36,.40),(.60,.47))
        until(lambda:browser.status().startswith('Refined selection locally · 1') or browser.error(),
              'first real positive stroke decoder',65)
        if browser.error():raise RuntimeError('Native stroke failed: '+browser.error())
        first_seconds=time.monotonic()-first
        one=browser.eval("document.querySelector('[aria-label=\"MagicTouch mask status\"]')?.textContent")
        print('POSITIVE_STROKE_SECONDS',round(first_seconds,2),'MASK',one,flush=True)
        assert '1 strokes' in one and 'confidence mask' in one
        image=browser.eval("""(() => {
          const canvas=document.querySelector('canvas[aria-label="MagicTouch confidence overlay"]');
          if(!canvas)return null;
          const c=canvas.getContext('2d');
          const data=c?.getImageData(0,0,canvas.width,canvas.height).data;
          let positive=0;
          if(data)for(let i=3;i<data.length;i+=4)if(data[i]>0)positive++;
          return {width:canvas.width,height:canvas.height,positive};
        })()""")
        print('MASK_PIXELS',image,flush=True)
        assert image and image['positive']>0,'Real native confidence mask missing'
        browser.eval("""(() => {
          const box=document.querySelector('[aria-label="MagicTouch brush mode"]');
          box.value='negative';box.dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        second=time.monotonic()
        browser.drag((.45,.45),(.62,.62))
        until(lambda:browser.status().startswith('Refined selection locally · 2') or browser.error(),
              'second real negative stroke decoder',65)
        if browser.error():raise RuntimeError('Native refinement failed: '+browser.error())
        two=browser.eval("document.querySelector('[aria-label=\"MagicTouch mask status\"]')?.textContent")
        print('NEGATIVE_STROKE_SECONDS',round(time.monotonic()-second,2),'MASK',two,flush=True)
        assert '2 strokes' in two and 'confidence mask' in two
        def selected_percent(text):
            match=re.search(r'(\d+(?:\.\d+)?)% selected above 0.5',text)
            if not match:raise RuntimeError('No numeric mask coverage: '+repr(text))
            return float(match.group(1))
        before,after=selected_percent(one),selected_percent(two)
        print('CONFIDENCE_SELECTION_DELTA_PCT',round(after-before,1),flush=True)
        assert after < before - 5, 'Negative stroke did not meaningfully reduce selected pixels'
        assert browser.click('Undo stroke')
        until(lambda:browser.status().startswith('Refined selection locally · 1') or browser.error(),
              'undo refinement',60)
        assert browser.click('Clear strokes')
        assert browser.eval("document.querySelector('[aria-label=\"MagicTouch mask status\"]').textContent.includes('0 strokes')")
        assert browser.click('Unload model')
        until(lambda:browser.status().startswith('Model unloaded'),'native unload',45)
        known_native_log_markers=(
          'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.',
          'gl_context.cc:408] GL version:',
          'gl_context.cc:1136] OpenGL error checking is disabled',
          'inference_feedback_manager.cc:121] Feedback manager requires',
          'gl_context_webgl.cc:149] Successfully destroyed WebGL context',
        )
        errors=[e for e in browser.errors if not any(marker in e for marker in known_native_log_markers)]
        print('BROWSER_ERRORS',errors[-6:],flush=True)
        assert not errors, 'Unexpected browser exceptions'
        print('PASS: pinned model, one image encoding, positive+negative repeat masks, undo, clear and native unload',flush=True)
    finally:
        if connection:connection.close()
        for proc in (chrome,preview):
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=4)
                except subprocess.TimeoutExpired:proc.kill()
        shutil.rmtree(folder,ignore_errors=True)

if __name__=='__main__':
    main()
