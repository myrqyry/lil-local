#!/usr/bin/env python3
"""Opt-in real Chromium qualification for Lil Local's packed Vision consumer.

Needs Chromium and Python websocket-client. Runs a disposable fake camera,
downloads eight official Vision task models only after explicit model actions,
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
            expected = ('classification heads' if task == 'classifier' else 'semantic classes' if task == 'semantic' else 'detected')
            until(lambda: self.eval("""(() => {
              const s=[...document.querySelectorAll('[aria-label="Vision detections"] p')][0];
              return !!s && s.textContent.includes(EXPECTED);
            })()""".replace('EXPECTED', json.dumps(expected))),
                  task + ' inference completed', 25)
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
          '!!document.querySelector(\'button[aria-label="Open Live Vision · Ten Local Processors"]\')'),
          'new Vision model card', 40)
        assert not any(name in u for u in browser.resources for name in ('face_landmarker.task', 'gesture_recognizer.task', 'pose_landmarker_lite.task', 'efficientdet_lite0.tflite', 'efficientnet_lite0.tflite', 'holistic_landmarker.task', 'deeplab_v3.tflite', 'blaze_face_short_range.tflite')), 'Early weights fetch'
        assert browser.click('Open Live Vision · Ten Local Processors')
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

        # Google's own Object Detector test uses a dog reference image.
        # Download only for the opt-in qualification run, at pinned source SHA.
        dog_image = Path(OUT) / 'dog.jpg'
        dog_url = ('https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples-web/'
          '6887f6399745dc549dadd0022c31fc9ab01f26d6/public/dog.jpg')
        with urllib.request.urlopen(dog_url, timeout=25) as response:
            dog_image.write_bytes(response.read())
        dog_digest='87295aa79a9e285f86a94c58230c50401453dbc8cca946d544ce0e3de3f71cf7'
        assert hashlib.sha256(dog_image.read_bytes()).hexdigest() == dog_digest, 'Upstream dog photo changed'

        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='object';select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'object'
        dog_dom=browser.send('DOM.getDocument')
        dog_node=browser.send('DOM.querySelector',{
          'nodeId':dog_dom['root']['nodeId'],
          'selector':'input[aria-label="Image for local vision analysis"]',
        })['nodeId']
        assert dog_node,'Object detector image picker missing'
        browser.send('DOM.setFileInputFiles', {'nodeId':dog_node,'files':[str(dog_image)]})
        browser.eval("""(() => {
          document.querySelector('input[aria-label="Image for local vision analysis"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda: browser.click('Analyze selected image locally'),
              'Analyze dog image', 15)
        until(lambda: browser.status()=='Image analyzed locally' or
              bool(browser.eval("document.querySelector('[role=alert]')?.textContent")),
              'Real dog object detection', 80)
        dog_error=browser.eval("document.querySelector('[role=alert]')?.textContent")
        if dog_error: raise AssertionError('Native object detection failed: '+dog_error)
        dog_summary=browser.eval(
          "document.querySelector('[aria-label=\"Vision detections\"]')?.textContent || ''")
        rectangles=browser.eval("document.querySelectorAll('svg rect').length")
        print('OBJECT_IMAGE',dog_summary,'BBOX_OVERLAYS',rectangles,flush=True)
        assert 'dog' in dog_summary.lower(), 'Official dog photo produced no dog detection'
        assert rectangles >= 1, 'Object detector pixel-space bounding boxes not rendered'
        assert any('efficientdet_lite0.tflite' in u for u in browser.resources), 'Official EfficientDet model missing'
        assert browser.eval("window.__visionTestTracks.length === 3"), 'Still-image object inference unexpectedly requested camera'

        assert browser.click('Start local vision')
        browser.wait_running('object')
        print('OBJECT_VIDEO',browser.status(),flush=True)
        browser.stop()

        # Whole-image category classification is distinct from bounding-box detection.
        # Reuse the same SHA-verified upstream dog image, without opening the camera.
        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='classifier';select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'classifier'
        assert browser.click('Analyze selected image locally')
        until(lambda: browser.status()=='Image analyzed locally' or
              bool(browser.eval("document.querySelector('[role=alert]')?.textContent")),
              'Real EfficientNet category classification', 80)
        classifier_error=browser.eval("document.querySelector('[role=alert]')?.textContent")
        if classifier_error:
            raise AssertionError('Native image classification failed: '+classifier_error)
        classifier_summary=browser.eval(
          "document.querySelector('[aria-label=\"Vision detections\"]')?.textContent || ''")
        print('CLASSIFIER_IMAGE',classifier_summary[:680],flush=True)
        assert any(word in classifier_summary.lower()
                   for word in ('dog','retriever','labrador','canine')), (
            'Official dog photo produced no expected image category')
        assert any('efficientnet_lite0.tflite' in u for u in browser.resources), (
            'Official EfficientNet-Lite0 model not loaded')
        assert browser.eval("window.__visionTestTracks.length === 4"), (
            'Still-image classification unexpectedly requested webcam')

        assert browser.click('Start local vision')
        browser.wait_running('classifier')
        print('CLASSIFIER_VIDEO',browser.status(),flush=True)
        browser.stop()

        # One actual native holistic session (face + pose + both-hand channels).
        # Use the upstream human image, independently pinned above, and no
        # camera permission until the separate VIDEO test.
        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='holistic';select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'holistic'
        holistic_dom=browser.send('DOM.getDocument')
        holistic_node=browser.send('DOM.querySelector',{
          'nodeId':holistic_dom['root']['nodeId'],
          'selector':'input[aria-label="Image for local vision analysis"]',
        })['nodeId']
        assert holistic_node, 'Holistic image selection input is missing'
        browser.send('DOM.setFileInputFiles',{
          'nodeId':holistic_node,'files':[str(pose_image)],
        })
        browser.eval("""(() => {
          document.querySelector('input[aria-label="Image for local vision analysis"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda: browser.click('Analyze selected image locally'),
              'Holistic still-image action',15)
        until(lambda: browser.status()=='Image analyzed locally' or
              bool(browser.eval("document.querySelector('[role=alert]')?.textContent")),
              'Real Holistic human photo inference',100)
        holistic_error=browser.eval("document.querySelector('[role=alert]')?.textContent")
        if holistic_error:
            raise AssertionError('Native Holistic inference failed: '+holistic_error)
        holistic_summary=browser.eval(
          "document.querySelector('[aria-label=\"Vision detections\"]')?.textContent || ''")
        holistic_dots=browser.eval("document.querySelectorAll('svg circle').length")
        print('HOLISTIC_IMAGE',holistic_summary[:550],'OVERLAY_DOTS',holistic_dots,flush=True)
        assert 'Holistic pose 1' in holistic_summary, (
            'Upstream human photo produced no holistic pose')
        assert '33 image / 33 world points' in holistic_summary, (
            'Holistic pose image/world coordinates unpaired')
        assert holistic_dots >= 33, 'Holistic pose overlay missing'
        assert any('holistic_landmarker.task' in u for u in browser.resources), (
            'Official Holistic model was not requested')
        assert browser.eval("window.__visionTestTracks.length === 5"), (
            'Holistic photo unexpectedly opened a camera')

        assert browser.click('Start local vision')
        browser.wait_running('holistic')
        print('HOLISTIC_VIDEO',browser.status(),flush=True)
        browser.stop()

        # Semantic segmentation returns per-pixel class IDs, not one object
        # bounding box nor a SAM foreground/alpha mask.
        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='semantic';select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'semantic'
        seg_doc=browser.send('DOM.getDocument')
        seg_node=browser.send('DOM.querySelector',{
          'nodeId':seg_doc['root']['nodeId'],
          'selector':'input[aria-label="Image for local vision analysis"]',
        })['nodeId']
        assert seg_node, 'Semantic mask image picker is missing'
        browser.send('DOM.setFileInputFiles', {
          'nodeId':seg_node,'files':[str(dog_image)],
        })
        browser.eval("""(() => {
          document.querySelector('input[aria-label="Image for local vision analysis"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda: browser.click('Analyze selected image locally'),
              'Semantic mask still-image action',15)
        until(lambda: browser.status()=='Image analyzed locally' or
              bool(browser.eval("document.querySelector('[role=alert]')?.textContent")),
              'Real DeepLab V3 dog-photo segmentation',100)
        semantic_error=browser.eval("document.querySelector('[role=alert]')?.textContent")
        if semantic_error: raise AssertionError('Native DeepLab segmentation failed: '+semantic_error)
        semantic_summary=browser.eval(
          "document.querySelector('[aria-label=\"Vision detections\"]')?.textContent || ''")
        semantic_canvas=browser.eval("""(() => {
          const canvas=document.querySelector('canvas[aria-label="Semantic segmentation class overlay"]');
          if(!canvas)return null;
          const ctx=canvas.getContext('2d');
          if(!ctx)return null;
          const rgba=ctx.getImageData(0,0,canvas.width,canvas.height).data;
          let colored=0;
          for(let i=3;i<rgba.length;i+=4) if(rgba[i]>0)colored++;
          return {width:canvas.width,height:canvas.height,colored};
        })()""")
        print('SEMANTIC_IMAGE',semantic_summary[:700], 'CANVAS',semantic_canvas,flush=True)
        assert 'semantic classes' in semantic_summary, 'Semantic output type missing'
        assert semantic_canvas and semantic_canvas['colored'] > 0, (
          'No actual class segmentation colors drawn')
        assert 'dog' in semantic_summary.lower(), (
          'Official dog reference image lacks expected dog semantic class')
        assert any('deeplab_v3.tflite' in u for u in browser.resources), (
          'DeepLab official model was not loaded')
        assert browser.eval("window.__visionTestTracks.length === 6"), (
          'Still-image semantic segmentation unexpectedly requested camera')

        assert browser.click('Start local vision')
        browser.wait_running('semantic')
        print('SEMANTIC_VIDEO',browser.status(),flush=True)
        browser.stop()

        # Short-range face detection needs a face-sized, SHA-pinned portrait,
        # not the upstream detector's dog photo or the distant full-body pose image.
        face_image=Path(OUT)/'face_model.png'
        face_url=('https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples-web/'
          '6887f6399745dc549dadd0022c31fc9ab01f26d6/public/face_model.png')
        with urllib.request.urlopen(face_url,timeout=25) as response:
            face_image.write_bytes(response.read())
        assert hashlib.sha256(face_image.read_bytes()).hexdigest()==(
            '75171e877e92b7a126cca2e7a388fc430225e07e9cd2e9e801eaa67ea6d7f4d9')
        assert browser.eval("""(() => {
          const select=document.querySelector('#vision-task');
          select.value='blazeFace';select.dispatchEvent(new Event('change',{bubbles:true}));
          return select.value;
        })()""") == 'blazeFace'
        blaze_dom=browser.send('DOM.getDocument')
        blaze_node=browser.send('DOM.querySelector',{
          'nodeId':blaze_dom['root']['nodeId'],
          'selector':'input[aria-label="Image for local vision analysis"]',
        })['nodeId']
        assert blaze_node, 'Face detector still-image input missing'
        browser.send('DOM.setFileInputFiles',{
          'nodeId':blaze_node,'files':[str(face_image)],
        })
        browser.eval("""(() => {
          document.querySelector('input[aria-label="Image for local vision analysis"]')
            .dispatchEvent(new Event('change',{bubbles:true}));
        })()""")
        until(lambda: browser.click('Analyze selected image locally'),
              'BlazeFace analyze action',15)
        until(lambda: browser.status()=='Image analyzed locally' or
              bool(browser.eval("document.querySelector('[role=alert]')?.textContent")),
              'Real human BlazeFace GPU detector',90)
        blaze_error=browser.eval("document.querySelector('[role=alert]')?.textContent")
        if blaze_error:raise AssertionError('Native BlazeFace failure: '+blaze_error)
        blaze_summary=browser.eval(
          "document.querySelector('[aria-label=\"Vision detections\"]')?.textContent || ''")
        blaze_boxes=browser.eval(
          "document.querySelectorAll('svg rect').length")
        blaze_marks=browser.eval(
          "document.querySelectorAll('svg circle').length")
        print('BLAZEFACE_IMAGE',blaze_summary[:520],
              'BOXES',blaze_boxes,'KEYPOINT_DOTS',blaze_marks,flush=True)
        assert 'Face 1' in blaze_summary, 'Human photo produced no face detection'
        assert 'keypoints' in blaze_summary, 'Face keypoint count missing'
        assert blaze_boxes>=1 and blaze_marks>=1, 'Face ROI/landmark overlay not drawn'
        assert any('blaze_face_short_range.tflite' in u for u in browser.resources), (
          'Publisher BlazeFace asset missing')
        assert browser.eval("window.__visionTestTracks.length === 7"), (
          'Still-image face detection unexpectedly opened webcam')
        assert browser.click('Start local vision')
        browser.wait_running('blazeFace')
        print('BLAZEFACE_VIDEO',browser.status(),flush=True)
        browser.stop()

        failures = [event for event in browser.errors
                    if 'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.' not in event]
        print('CONSOLE_EVENTS',browser.errors[-4:],flush=True)
        assert not failures, 'Browser runtime exceptions: '+repr(failures)
        print('PASS: 8 SHA-verified models; real human face ROI and keypoints, pose/holistic, dog inference, semantic mask; all eight video paths; camera cleanup',flush=True)
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
