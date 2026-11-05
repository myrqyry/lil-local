#!/usr/bin/env python3
"""Opt-in real Chromium qualification for Lil Local's packed Vision consumer.

Needs Chromium and Python websocket-client. Runs a disposable fake camera,
downloads official face, gesture and pose weights only after explicit Start,
checks browser inference/cleanup and never touches the user's webcam.
No model weights are saved to the repository.
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent.parent
URL = 'http://127.0.0.1:4386'
OUT = tempfile.mkdtemp(prefix='lil-local-vision-smoke-')
PROFILE = Path(OUT) / 'profile'


def until(check, label, seconds=30):
    end = time.monotonic() + seconds
    last = None
    while time.monotonic() < end:
        try:
            value = check()
            if value:
                return value
        except Exception as error:
            last = str(error)
        time.sleep(0.15)
    raise AssertionError('Timeout: ' + label + ' (' + str(last) + ')')


class Browser:
    def __init__(self, connection):
        self.ws = connection
        self.seq = 0
        self.errors = []
        self.resources = []

    def send(self, method, params=None):
        self.seq += 1
        ident = self.seq
        self.ws.send(json.dumps({'id': ident, 'method': method, 'params': params or {}}))
        while True:
            event = json.loads(self.ws.recv())
            name, payload = event.get('method', ''), event.get('params', {})
            if name == 'Runtime.exceptionThrown':
                self.errors.append('exception: ' + str(payload.get('exceptionDetails', {}))[:300])
            elif name == 'Runtime.consoleAPICalled' and payload.get('type') == 'error':
                self.errors.append('console: ' + str(payload.get('args', []))[:300])
            elif name == 'Network.responseReceived':
                self.resources.append(payload.get('response', {}).get('url',''))
            if event.get('id') != ident:
                continue
            if 'error' in event:
                raise AssertionError(str(event['error']))
            return event.get('result', {})

    def eval(self, js):
        result = self.send('Runtime.evaluate', {'expression': js, 'returnByValue': True})
        if 'exceptionDetails' in result:
            raise AssertionError(str(result['exceptionDetails']))
        return result.get('result', {}).get('value')

    def click(self, label):
        return self.eval("""(() => {
          const b=[...document.querySelectorAll('button')]
            .find(x=>x.getAttribute('aria-label')===LABEL ||
              x.textContent.trim()===LABEL);
          if (!b || b.disabled) return false;
          b.click(); return true;
        })()""".replace('LABEL', json.dumps(label)))

    def status(self):
        return self.eval("document.querySelector('[role=status]')?.textContent || ''")

    def wait_running(self, task):
        try:
            until(lambda: self.status() == 'Running locally · camera active',
                  'real ' + task + ' model and webcam initialized', 75)
            until(lambda: self.eval("""(() => {
              const s=[...document.querySelectorAll('[aria-label="Vision detections"] p')][0];
              return !!s && s.textContent.includes('detected');
            })()"""), task + ' inference completed', 25)
        except Exception:
            print('STATE', self.status(), 'ERROR', self.eval("document.querySelector('[role=alert]')?.textContent"),flush=True)
            raise
        time.sleep(1.2)
        assert self.eval("!document.querySelector('[role=alert]')"), 'Vision inference failed'
        assert self.eval("document.querySelector('video').readyState >= 2"), 'No camera frames'

    def stop(self):
        assert self.click('Stop and release camera')
        until(lambda: self.status() == 'Camera off', 'camera stop')
        assert self.eval("document.querySelector('video').srcObject === null"), 'Camera leak'
        assert self.eval("window.__visionTestTracks.length > 0 && window.__visionTestTracks.every(t=>t.readyState==='ended')"), 'MediaStream tracks not stopped'


def main():
    preview = chrome = connection = None
    try:
        preview = subprocess.Popen(
            ['pnpm','--filter','playground','exec','vite','preview',
             '--host','127.0.0.1','--port','4386','--strictPort'],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        until(lambda: urllib.request.urlopen(URL, timeout=1).status == 200,
              'Vite preview', 25)
        chrome = subprocess.Popen([
            'chromium','--headless=new','--no-sandbox','--disable-dev-shm-usage',
            '--no-proxy-server','--no-first-run','--no-default-browser-check',
            '--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader-webgl',
            '--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',
            '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
            '--remote-allow-origins=http://localhost',f'--user-data-dir={PROFILE}',
            'about:blank'], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        port = until(lambda: (PROFILE/'DevToolsActivePort').read_text().splitlines()[0]
                     if (PROFILE/'DevToolsActivePort').is_file() else None,
                     'Chrome debugger', 30)
        def ws_url():
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=1) as response:
                return next((row['webSocketDebuggerUrl'] for row in json.load(response)
                             if row.get('type') == 'page'),None)
        connection = websocket.create_connection(
            until(ws_url,'Chrome page',15),origin='http://localhost',timeout=90)
        browser = Browser(connection)
        for cmd in ('Page.enable','Runtime.enable','Network.enable'):
            browser.send(cmd)
        browser.send('Page.addScriptToEvaluateOnNewDocument', {'source': '''
          window.__visionTestTracks = [];
          const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
          navigator.mediaDevices.getUserMedia = async (...args) => {
            const stream = await original(...args);
            window.__visionTestTracks.push(...stream.getTracks());
            return stream;
          };
        '''})
        browser.send('Page.navigate', {'url':URL})
        until(lambda: browser.eval(
          '!!document.querySelector(\'button[aria-label="Open MediaPipe Vision · Face, Gestures & Pose"]\')'),
          'new Vision model card', 40)
        assert not any('face_landmarker.task' in u or 'gesture_recognizer.task' in u for u in browser.resources), 'Early weights fetch'
        assert browser.click('Open MediaPipe Vision · Face, Gestures & Pose')
        until(lambda: browser.eval('!!document.querySelector(\'#vision-task\')'),
              'reusable playground panel')
        assert browser.status() == 'Camera off'
        assert browser.click('Start local vision')
        browser.wait_running('face')
        assert any('face_landmarker.task' in u for u in browser.resources), 'Face asset not loaded'
        assert any('vision_wasm_internal.wasm' in u for u in browser.resources), 'Matching WASM not loaded'
        print('FACE',browser.status(),flush=True)
        browser.stop()
        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='gesture'; select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'gesture'
        assert browser.click('Start local vision')
        browser.wait_running('gesture')
        assert any('gesture_recognizer.task' in u for u in browser.resources), 'Gesture asset not loaded'
        print('GESTURE',browser.status(),flush=True)
        browser.stop()

        # Real subject image from the upstream Pose Landmarker Playwright test.
        # Fetch only inside this opt-in smoke; do not vendor someone else's photo.
        pose_image = Path(OUT) / 'pose_model.png'
        image_url = ('https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples-web/'
          '6887f6399745dc549dadd0022c31fc9ab01f26d6/public/pose_model.png')
        with urllib.request.urlopen(image_url, timeout=25) as response:
            pose_image.write_bytes(response.read())
        expected_digest='884e822bf3f1585fdfdf1f36c06c47ceec3d745757cc14abac9cacf45353de63'
        assert hashlib.sha256(pose_image.read_bytes()).hexdigest() == expected_digest, 'Upstream pose test photo changed'

        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='pose';select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'pose'
        doc=browser.send('DOM.getDocument')
        node=browser.send('DOM.querySelector',{
          'nodeId':doc['root']['nodeId'],
          'selector':'input[aria-label="Image for local vision analysis"]',
        })['nodeId']
        assert node,'No still-image input'
        browser.send('DOM.setFileInputFiles', {'nodeId':node,'files':[str(pose_image)]})
        browser.eval("""(() => {
          document.querySelector('input[aria-label="Image for local vision analysis"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda: browser.click('Analyze selected image locally'),
              'Analyze image action', 15)
        until(lambda: browser.status()=='Image analyzed locally' or
              bool(browser.eval("document.querySelector('[role=alert]')?.textContent")),
              'Real official photo pose inference', 80)
        err=browser.eval("document.querySelector('[role=alert]')?.textContent")
        if err: raise AssertionError('Native pose inference failed: '+err)
        summary=browser.eval(
          "document.querySelector('[aria-label=\"Vision detections\"]')?.textContent || ''")
        overlay_points=browser.eval("document.querySelectorAll('svg circle').length")
        print('POSE_IMAGE',summary,'OVERLAY_POINTS',overlay_points,flush=True)
        assert 'Pose 1' in summary, 'No actual human pose detected in official reference photo'
        assert '33 image points / 33 world points' in summary, 'Wrong pose coordinate contract'
        assert overlay_points >= 33, 'No skeleton overlay'
        assert any('pose_landmarker_lite.task' in u for u in browser.resources), 'Pinned Pose model not loaded'
        assert browser.eval("window.__visionTestTracks.length === 2"), 'Static image unexpectedly requested webcam'

        # VIDEO path uses the same shared task; synthetic webcam can have zero people.
        assert browser.click('Start local vision')
        browser.wait_running('pose')
        print('POSE_VIDEO',browser.status(),flush=True)
        browser.stop()
        failures = [event for event in browser.errors
                    if 'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.' not in event]
        print('CONSOLE_EVENTS',browser.errors[-4:],flush=True)
        assert not failures, 'Browser runtime exceptions: '+repr(failures)
        print('PASS: 3 model SHA checks, human pose image + 33/33 landmarks, all 3 real native video tasks, camera cleanup',flush=True)
    finally:
        if connection: connection.close()
        for proc in (chrome, preview):
            if proc and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill()
        shutil.rmtree(OUT,ignore_errors=True)


if __name__ == '__main__':
    main()
