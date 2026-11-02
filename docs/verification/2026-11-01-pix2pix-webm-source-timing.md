# Pix2Pix WebM source-timing correction — 2026-11-01

## Failure mechanism

The experimental video export previously captured a canvas stream with `MediaRecorder` while processing each input video frame sequentially. `MediaRecorder` uses wall-clock capture timestamps, meaning a frame that took more than 125 ms to process at target 8 fps could stretch the exported clip duration. The requested 8 fps was therefore not a reliable output timeline.

## Change

Replaced only the export writer with `mediabunny@1.65.0` `CanvasSource` and `WebMOutputFormat`. `CanvasSource.add(sourceTimeSeconds, frameDurationSeconds)` supplies explicit source-derived timestamps independent of wall-clock inference latency. The existing source-frame seeking, Pix2Pix Rust/Wasm WebGPU inference, OPFS model storage, preview path and cancellation remain intact. VP9 is preferred, VP8 is fallback, and browsers without working WebCodecs support fail visibly. The export is capped at 60 seconds of source video at 8 fps, and the output remains video-only. An interrupted writer is cancelled and the partial file is not presented for download.

## Verification

- Two new time-boundary unit tests demonstrated failures on the old implementation and passed with the correction.
- `pnpm --filter playground typecheck` passed; full `pnpm verify` exited 0 after final cleanup edits (unit, package, qualification, packed consumer, and build gates).
- Real Chromium headless smoke on `yuzu-mint`: used the installed Mediabunny package through Vite, generated eight alternating canvas frames at source times 0, 0.125, ..., 0.875 seconds, delayed **170 ms per frame** (slower than real-time 8 fps), and encoded VP8 WebM. Browser page errors: 0. `ffprobe` reported `duration=1.000000`, `size=1107` bytes on `/tmp/lil-local-webm-timing-smoke.webm`. This validates the browser writer timing mechanism, not the full model-to-video pipeline.

## Remaining limitations

- No actual Pix2Pix full video export was exercised against the pinned weights in this change, so end-to-end visual correctness and GPU-canvas capture need separate proof.
- Encoder availability is browser-dependent; BufferTarget assembles the whole output in memory, hence the conservative 60-second limit; 60 seconds is not a proven safe memory bound for all devices.
- Source frames are decoded using HTMLVideoElement seeks and `seeked`, not a frame-accurate WebCodecs decoder, so frame-selection accuracy still needs cross-browser evaluation.
- Audio is not preserved. The new timing test is a browser codec/mux proof, not model throughput benchmark.
