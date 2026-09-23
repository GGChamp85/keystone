// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Incremental decoder for the text/event-stream frames GET /v1/keystone/tasks/{id}/stream sends
// (src/api/routes/agents.py): `id: <redis stream id>\ndata: <json>\n\n` per event, and the SSE
// comment `: keep-alive\n\n` while the route waits for the next event. Exactly the split
// web/src/api.ts's streamTask does inline — frames end at a blank line, the payload is the one
// `data: ` line, a frame with no data line (a comment) is skipped — factored out so the extension
// can feed it recorded real frames in tests, chunked at arbitrary byte boundaries.

const DATA_PREFIX = 'data: '

export class SseFrameDecoder {
  private buffer = ''
  private readonly decoder = new TextDecoder()

  /** Feed the next chunk of the response body; returns the `data:` payloads of every frame completed by it. */
  push(chunk: Uint8Array | string): string[] {
    this.buffer += typeof chunk === 'string' ? chunk : this.decoder.decode(chunk, { stream: true })
    const frames = this.buffer.split('\n\n')
    this.buffer = frames.pop() ?? ''
    const payloads: string[] = []
    for (const frame of frames) {
      const dataLine = frame.split('\n').find((line) => line.startsWith(DATA_PREFIX))
      if (!dataLine) continue // keep-alive comment lines (": keep-alive") have no "data: " prefix
      payloads.push(dataLine.slice(DATA_PREFIX.length))
    }
    return payloads
  }

  /** Bytes received after the last complete frame — non-empty only if the stream ended mid-frame. */
  get pending(): string {
    return this.buffer
  }
}

/** JSON.parse that skips a malformed payload rather than crashing the whole stream (same as web/src/api.ts). */
export function parsePayloads<T>(payloads: string[]): T[] {
  const out: T[] = []
  for (const raw of payloads) {
    try {
      out.push(JSON.parse(raw) as T)
    } catch {
      // malformed frame — skip
    }
  }
  return out
}
