/* Adapted from People-11/Nemotron_Web (MIT); see LICENSE.upstream.txt. */
import type { MelSpectrogram } from "./audio";
import type { ChunkProgress, StreamProgress } from "./nemotron";
import type { NemotronAssets } from "./model";
export type { ChunkProgress, StreamProgress } from "./nemotron";

// Same API surface as NemotronBrowserASR, but the model runs in a worker so heavy
// inference (mel + WebGPU readbacks) never blocks the main thread / starves rAF.

export type WorkerRequest =
  | { id: number; type: "load"; backend: "auto" | "webgpu" | "wasm"; assets?: NemotronAssets }
  | { id: number; type: "dispose" }
  | { id: number; type: "transcribe"; mel: MelSpectrogram; langId: number }
  | { id: number; type: "startStream"; langId: number }
  | { id: number; type: "pushAudio"; samples: ArrayBuffer; flush: boolean }
  | { type: "abort" };

export type WorkerResponse =
  | { id: number; type: "status"; message: string }
  | { id: number; type: "chunk"; progress: ChunkProgress }
  | { id: number; type: "partial"; text: string }
  | { id: number; type: "result"; value: unknown }
  | { id: number; type: "error"; message: string };

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onStatus?: (message: string) => void;
  onChunk?: (progress: ChunkProgress) => void;
  onPartial?: (text: string) => void;
};

export interface NemotronClientOptions {
  /** Overrides the standard ES module worker, useful for a host worker pool. */
  workerFactory?: () => Worker;
  /** Optional model and local argmax graph paths, supplied by the host app. */
  assets?: NemotronAssets;
}

export class NemotronClient {
  private worker: Worker;
  private readonly assets: NemotronAssets | undefined;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor(options: NemotronClientOptions = {}) {
    this.assets = options.assets;
    this.worker = options.workerFactory?.() ?? new Worker(new URL("./nemotron.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      const entry = this.pending.get(data.id);
      if (!entry) return;
      if (data.type === "status") entry.onStatus?.(data.message);
      else if (data.type === "chunk") entry.onChunk?.(data.progress);
      else if (data.type === "partial") entry.onPartial?.(data.text);
      else if (data.type === "result") {
        this.pending.delete(data.id);
        entry.resolve(data.value);
      } else {
        this.pending.delete(data.id);
        entry.reject(new Error(data.message));
      }
    };
  }

  private request<T>(
    message: Extract<WorkerRequest, { id: number }>,
    transfer: Transferable[] = [],
    handlers: Pick<Pending, "onStatus" | "onChunk" | "onPartial"> = {},
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(message.id, { resolve: resolve as Pending["resolve"], reject, ...handlers });
      this.worker.postMessage(message, transfer);
    });
  }

  load(onStatus: (message: string) => void, backend: "auto" | "webgpu" | "wasm" = "auto"): Promise<void> {
    return this.request<void>({ id: this.nextId++, type: "load", backend, assets: this.assets }, [], { onStatus });
  }

  async dispose(): Promise<void> {
    try { await this.request<void>({ id: this.nextId++, type: "dispose" }); }
    finally { this.worker.terminate(); }
  }

  transcribe(
    mel: MelSpectrogram,
    langId: number,
    onChunk: (progress: ChunkProgress) => void,
    signal?: AbortSignal,
  ): Promise<string> {
    // Clone mel (no transfer) so the caller can re-run the same file.
    signal?.addEventListener("abort", () => this.worker.postMessage({ type: "abort" }), { once: true });
    return this.request<string>({ id: this.nextId++, type: "transcribe", mel, langId }, [], { onChunk });
  }

  startStream(langId: number): void {
    // Fire-and-forget: the worker handles requests in postMessage order, so this
    // resets stream state before any pushAudio that follows it.
    void this.request<void>({ id: this.nextId++, type: "startStream", langId }).catch(() => {});
  }

  pushAudio(samples: Float32Array, flush = false, onPartial?: (text: string) => void): Promise<StreamProgress> {
    // Never transfer the caller's own buffer; Web Audio may retain or reuse it.
    const buffer = new Float32Array(samples).buffer;
    return this.request<StreamProgress>(
      { id: this.nextId++, type: "pushAudio", samples: buffer, flush },
      [buffer],
      { onPartial },
    );
  }
}
