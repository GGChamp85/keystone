// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The live trace panel: one webview per task, fed by KeystoneClient.streamTask (the fetch-based
// SSE reader) from the extension host. The host keeps the event list and describes each step
// (trace/render.ts); the webview (media/trace.js) only draws what it is sent, so nothing in the
// webview is trusted with the API key or a network connection. The webview asks for a snapshot
// when its script has loaded, so events that arrived before it was ready are never lost.

import * as vscode from 'vscode'
import { isNodeEvent, isStepEvent, type KeystoneClient, type NodeEvent, type TaskEvent } from '../api'
import { describeStep, type DescribedStep, summarize, type TraceSummary } from './render'

export interface RenderedEvent {
  event: TaskEvent
  step?: DescribedStep
}

export type HostMessage =
  | { type: 'snapshot'; taskId: string; items: RenderedEvent[]; summary: TraceSummary; connected: boolean; error: string | null }
  | { type: 'event'; item: RenderedEvent; summary: TraceSummary }
  | { type: 'status'; connected: boolean; error: string | null; summary: TraceSummary }

export class TracePanel implements vscode.Disposable {
  private static readonly openPanels = new Map<string, TracePanel>()

  /** Show the trace for `taskId`, reusing the panel if it is already open. */
  static show(extensionUri: vscode.Uri, client: KeystoneClient, taskId: string): TracePanel {
    const existing = TracePanel.openPanels.get(taskId)
    if (existing) {
      existing.panel.reveal()
      return existing
    }
    const panel = vscode.window.createWebviewPanel(
      'keystone.trace',
      `Keystone task ${taskId.slice(0, 8)}`,
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
      },
    )
    const trace = new TracePanel(panel, extensionUri, client, taskId)
    TracePanel.openPanels.set(taskId, trace)
    return trace
  }

  static get(taskId: string): TracePanel | undefined {
    return TracePanel.openPanels.get(taskId)
  }

  readonly events: TaskEvent[] = []
  private readonly eventEmitter = new vscode.EventEmitter<TaskEvent>()
  /** Fires for every event received from the stream — what the integration test listens to. */
  readonly onDidReceiveEvent = this.eventEmitter.event
  private readonly finishEmitter = new vscode.EventEmitter<TraceSummary>()
  /** Fires once, when the stream closes (after the engine's final event) or fails. */
  readonly onDidFinish = this.finishEmitter.event

  private connected = true
  private error: string | null = null
  private webviewReady = false
  private readonly abort = new AbortController()
  private readonly disposables: vscode.Disposable[] = []

  private constructor(
    private readonly panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri,
    private readonly client: KeystoneClient,
    readonly taskId: string,
  ) {
    panel.webview.html = this.html(extensionUri)
    this.disposables.push(
      panel.webview.onDidReceiveMessage((msg: { type?: string }) => {
        if (msg?.type === 'ready') {
          this.webviewReady = true
          this.postSnapshot()
        }
      }),
      panel.onDidDispose(() => this.dispose()),
    )
    void this.stream()
  }

  get isConnected(): boolean {
    return this.connected
  }

  /** Resolve once at least `count` events have arrived (events already received count), or reject on timeout. */
  waitForEvents(count: number, timeoutMs: number): Promise<TaskEvent[]> {
    if (this.events.length >= count) return Promise.resolve([...this.events])
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        sub.dispose()
        reject(new Error(`trace panel received ${this.events.length} event(s) in ${timeoutMs} ms, expected ${count}`))
      }, timeoutMs)
      const sub = this.onDidReceiveEvent(() => {
        if (this.events.length >= count) {
          clearTimeout(timer)
          sub.dispose()
          resolve([...this.events])
        }
      })
    })
  }

  private async stream(): Promise<void> {
    try {
      await this.client.streamTask(this.taskId, (ev) => this.handle(ev), this.abort.signal)
    } catch (err) {
      if (!this.abort.signal.aborted) this.error = err instanceof Error ? err.message : String(err)
    } finally {
      this.connected = false
      this.postStatus()
      this.finishEmitter.fire(summarize(this.events, false))
    }
  }

  private handle(ev: TaskEvent): void {
    this.events.push(ev)
    this.post({ type: 'event', item: this.render(ev), summary: summarize(this.events, this.connected) })
    this.eventEmitter.fire(ev)
    if (isNodeEvent(ev) && ev.final) this.notifyFinal(ev)
  }

  private render(ev: TaskEvent): RenderedEvent {
    return isStepEvent(ev) ? { event: ev, step: describeStep(ev) } : { event: ev }
  }

  private notifyFinal(ev: NodeEvent): void {
    const short = this.taskId.slice(0, 8)
    if (ev.pr_url) {
      const prUrl = ev.pr_url
      void vscode.window
        .showInformationMessage(`Keystone task ${short} complete — pull request #${ev.pr_number ?? '?'} opened.`, 'Open PR')
        .then((choice) => {
          if (choice === 'Open PR') void vscode.env.openExternal(vscode.Uri.parse(prUrl))
        })
    } else if (ev.phase === 'complete') {
      void vscode.window.showInformationMessage(`Keystone task ${short} complete. ${ev.result_summary ?? ''}`.trim())
    } else {
      void vscode.window.showErrorMessage(`Keystone task ${short} ${ev.phase}: ${ev.error_message ?? 'no error message'}`)
    }
  }

  private postSnapshot(): void {
    this.post({
      type: 'snapshot',
      taskId: this.taskId,
      items: this.events.map((ev) => this.render(ev)),
      summary: summarize(this.events, this.connected),
      connected: this.connected,
      error: this.error,
    })
  }

  private postStatus(): void {
    this.post({ type: 'status', connected: this.connected, error: this.error, summary: summarize(this.events, this.connected) })
  }

  private post(msg: HostMessage): void {
    if (!this.webviewReady) return // the webview asks for a snapshot once its script runs
    void this.panel.webview.postMessage(msg)
  }

  private html(extensionUri: vscode.Uri): string {
    const webview = this.panel.webview
    const script = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'media', 'trace.js'))
    const style = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'media', 'trace.css'))
    const nonce = Array.from({ length: 32 }, () => 'abcdefghijklmnopqrstuvwxyz0123456789'[Math.floor(Math.random() * 36)]).join('')
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${style}">
<title>Keystone task ${this.taskId}</title>
</head>
<body>
<main id="app" data-task-id="${this.taskId}">
  <p class="muted">Connecting to the task stream…</p>
</main>
<script nonce="${nonce}" src="${script}"></script>
</body>
</html>`
  }

  dispose(): void {
    if (!TracePanel.openPanels.has(this.taskId)) return
    TracePanel.openPanels.delete(this.taskId)
    this.abort.abort()
    for (const d of this.disposables) d.dispose()
    this.eventEmitter.dispose()
    this.finishEmitter.dispose()
    this.panel.dispose()
  }
}
