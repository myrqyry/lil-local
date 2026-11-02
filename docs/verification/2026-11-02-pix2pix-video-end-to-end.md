# Real Pix2Pix video export qualification — 2026-11-02

## Environment and scope

On yuzu-mint, Lil Local `master` after `afc22c2`, using a headed Chromium Playwright session with `/usr/bin/google-chrome-beta` (Chrome Beta 158), X11, experimental WebGPU flags, existing pinned folded `edges2shoes-batchnorm-eval-v1.bin` (217,677,836 bytes) imported into a fresh browser origin's OPFS, and the actual Pix2Pix panel's **Load installed model → Export full clip → Download WebM** UI flow. No user project files were cleared, and RikkaHub was not touched. This test did not collect a verified GPU adapter identity; historical NVIDIA/Turing full-image inference qualification is separate.

Generated `/tmp/lil-pix2pix-source-8frames.mp4` with FFmpeg `testsrc2=size=256x256:rate=8:duration=1`, H.264/yuv420p, exactly 8 decoded frames and 1 second duration. Output `/tmp/lil-pix2pix-real-e2e.webm` is test-only.

## Red: two genuine end-to-end failures

1. The initial real export hung at `Reading local video metadata for WebM export…` for the full 180-second Playwright timeout even though the video element had loaded the source and the model engine was ready. `await output.getMimeType()` ran **before any video frames were added**, while Mediabunny waits for codec output metadata. Moved MIME inspection until after `output.finalize()`.
2. Once unblocked, the browser immediately reported `OperationError: Encoding error. (Can't readback frame textures.)` because `CanvasSource` was encoding directly from the WebGPU output canvas. The Pix2Pix engine already returns the result as verified 256x256 RGBA pixels, so a separate 2D encoding canvas is now updated using the returned `ImageData` before each timestamped `source.add()`. No changes were made to the inference engine, weights or OPFS storage.

## Green: actual model-to-WebM export

The complete UI path succeeded after both corrections. Browser report: `Video export ready · 8 frames · 0.1 MiB WebM.` The Playwright run downloaded `lil-pix2pix-source-8frames-pix2pix-8fps.webm`, 63,180 bytes, with no reported page errors. `ffprobe -count_frames` reported:

```text
codec_name=vp9
width=256
height=256
avg_frame_rate=8/1
nb_read_frames=8
format.duration=1.000000
format.size=63180
```

FFmpeg decoded all 8 video frames and generated 8 distinct frame MD5 hashes. All 8 source frames also had distinct frame hashes; every exported frame hash differed from the corresponding source hash, though that alone does not establish semantic transformation quality. The WebM is demonstrably nonempty, decodable, correct in counted frames and source-duration timing. **No numerical model-output parity test was rerun** as part of this video export.

## Remaining boundaries

The test is one 1-second 8 fps local clip in one headed browser/GPU setup. Frame-selection accuracy across unusual codecs and variable-frame-rate inputs remains unqualified. Cancellation and repeated video exports still require real browser coverage; existing unit tests cover some lifecycle sequencing. The export is video-only and limited to 60s, using BufferTarget memory storage. Separate codec/browser environments have not been tested. Do not generalize one successful run into all-device support.
