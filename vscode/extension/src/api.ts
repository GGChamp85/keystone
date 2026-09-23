// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The Keystone REST client the extension host uses. web/src/api.ts is the same-origin SPA client;
// this one takes a base URL and the API key from VS Code's secret storage — otherwise the request
// shapes and the fetch-based SSE reader are the same, and the event declarations below (NodeEvent,
// StepEventType, StepEvent, TaskEvent) are kept byte-for-byte identical to web/src/api.ts by
// scripts/check_ts_types_in_sync.py (tests/test_ts_types_in_sync.py, CI's lint job).

import { parsePayloads, SseFrameDecoder } from './sse'

export interface KeystoneConnection {
  baseUrl: string // e.g. http://localhost:8080 — no trailing /v1
  apiKey: string
}

export class KeystoneApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'KeystoneApiError'
  }
}

export interface TaskSubmitRequest {
  task: string
  repository_url?: string
  branch?: string
  model?: string
  max_iterations?: number
}

export interface TaskSubmitResponse {
  task_id: string
  status: string
  message: string
}

// ── Memory (src/memory/store.py, src/api/routes/memory.py) ─────────

export interface MemoryRecord {
  id: string
  repository: string | null
  scope: string
  kind: string
  content: string
  source: string
  status: string
  pinned: boolean
  confidence: number
  hit_count: number
  created_by: string | null
}

// ── Team task list (src/api/routes/agents.py's GET /v1/keystone/tasks,
// src/api/models/responses.py's AgentTaskSummaryResponse) ──────────

export interface AgentTaskSummary {
  id: string
  user_id: string | null
  status: string
  task_description: string
  repository_url: string | null
  branch: string
  branch_name: string | null
  pr_url: string | null
  pr_number: number | null
  model_role: string
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

export interface ListTasksParams {
  user_id?: string
  repository_url?: string
  limit?: number
}

// Mirrors src/orchestrator/events.py exactly — keep these in sync.
// A node event is the per-node summary (_summarize); the engine's last one carries
// `final: true` plus the git results (publish_task_final).
export interface NodeEvent {
  event_type?: 'node' // absent on events recorded before step events existed
  node: string
  phase: string
  final?: boolean
  iteration?: number
  max_iterations?: number | null
  plan?: string
  plan_steps?: string[]
  current_plan_step?: number
  files_changed?: string[]
  review_passed?: boolean | null
  review_comment_count?: number
  tests_passed?: boolean | null
  test_pass_count?: number
  test_total_count?: number
  total_tokens?: number
  result_summary?: string
  error_message?: string | null
  branch_name?: string | null
  commit_sha?: string | null
  pr_url?: string | null
  pr_number?: number | null
  timestamp: number
}

// What happened INSIDE a node, as it happened, with the full payload (STEP_EVENT_TYPES).
export type StepEventType =
  | 'tool_call'
  | 'tool_result'
  | 'model_text'
  | 'test_output'
  | 'quality_findings'
  | 'deps_install'
  | 'repo_map'
  | 'route_decision'
  | 'diff'
  | 'pr'

export interface StepEvent {
  event_type: StepEventType
  node: string
  phase: string
  timestamp: number
  [key: string]: unknown
}

export type TaskEvent = NodeEvent | StepEvent

export function isStepEvent(ev: TaskEvent): ev is StepEvent {
  return typeof ev.event_type === 'string' && ev.event_type !== 'node'
}

export function isNodeEvent(ev: TaskEvent): ev is NodeEvent {
  return !isStepEvent(ev)
}

export const TERMINAL_PHASES = new Set(['complete', 'failed', 'cancelled'])

// ── Model Library (src/api/routes/models_library.py, src/inference/library.py) ─────────

export type ModelState = 'READY' | 'UNHEALTHY' | 'NOT_DEPLOYED' | 'TRAINING'

export interface LibraryAdapter {
  name: string
  base_model_id: string
  path: string
  rank: number
  job_type: string
  job_id: string | null
  status: string
  is_default: boolean
  metrics: Record<string, unknown>
  created_at: string | null
}

export interface LibraryModel {
  kind: 'role' | 'adapter' | 'catalog'
  id: string
  name: string
  purpose: string
  hf_source: string
  provider: string
  endpoint: string | null
  served_path: string | null
  served_model_ids?: string[]
  state: ModelState
  health?: Record<string, unknown>
  params_b?: number | null
  context?: number | null
  context_source?: string | null
  license?: string | null
  moe?: boolean | null
  tool_parser?: string | null
  deployment?: Record<string, number | string | null>
  api_features: Record<string, boolean | null>
  adapters?: LibraryAdapter[]
  adapter_status?: string
  is_default?: boolean
  path?: string
  rank?: number
  job_id?: string | null
  metrics?: Record<string, unknown>
}

export interface ModelLibrary {
  generated_at: number
  models: LibraryModel[]
}

export class KeystoneClient {
  constructor(
    private readonly conn: KeystoneConnection,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}

  get baseUrl(): string {
    return this.conn.baseUrl.replace(/\/+$/, '')
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`
  }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    return this.conn.apiKey ? { Authorization: `Bearer ${this.conn.apiKey}`, ...extra } : extra
  }

  private async fail(what: string, resp: Response): Promise<never> {
    const body = await resp.text().catch(() => '')
    throw new KeystoneApiError(`${what} (${resp.status})${body ? `: ${body}` : ''}`, resp.status)
  }

  async submitTask(req: TaskSubmitRequest): Promise<TaskSubmitResponse> {
    const resp = await this.fetchImpl(this.url('/v1/keystone/tasks'), {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(req),
    })
    if (!resp.ok) return this.fail('Task submission failed', resp)
    return (await resp.json()) as TaskSubmitResponse
  }

  async listTasks(params: ListTasksParams = {}): Promise<AgentTaskSummary[]> {
    const query = new URLSearchParams()
    if (params.user_id) query.set('user_id', params.user_id)
    if (params.repository_url) query.set('repository_url', params.repository_url)
    if (params.limit) query.set('limit', String(params.limit))
    const qs = query.toString()
    const resp = await this.fetchImpl(this.url(`/v1/keystone/tasks${qs ? `?${qs}` : ''}`), { headers: this.headers() })
    if (!resp.ok) return this.fail('Failed to list tasks', resp)
    return (await resp.json()) as AgentTaskSummary[]
  }

  async getModelLibrary(): Promise<ModelLibrary> {
    const resp = await this.fetchImpl(this.url('/v1/keystone/models'), { headers: this.headers() })
    if (!resp.ok) return this.fail('Failed to load the model library', resp)
    return (await resp.json()) as ModelLibrary
  }

  /**
   * POST /v1/keystone/memory/recall — exactly what the agent itself would recall for this query
   * (the same ranked recall src/orchestrator/nodes/planning.py and coding.py use), pinned first,
   * repo-scope over tenant-scope, then keyword relevance and recency.
   */
  async recallMemories(query: string, repository?: string): Promise<MemoryRecord[]> {
    const resp = await this.fetchImpl(this.url('/v1/keystone/memory/recall'), {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ query, repository: repository || null }),
    })
    if (!resp.ok) return this.fail('Memory recall failed', resp)
    return (await resp.json()) as MemoryRecord[]
  }

  /**
   * Reads GET /v1/keystone/tasks/{id}/stream as Server-Sent Events using `fetch` + a manual
   * ReadableStream reader rather than a native EventSource — EventSource cannot set an
   * Authorization header, and this project's whole API is Bearer-token authenticated (the same
   * reader web/src/api.ts uses). Resolves when the server closes the stream (after the engine's
   * `final` event) or the signal aborts.
   */
  async streamTask(taskId: string, onEvent: (ev: TaskEvent) => void, signal: AbortSignal): Promise<void> {
    const resp = await this.fetchImpl(this.url(`/v1/keystone/tasks/${taskId}/stream`), {
      headers: this.headers(),
      signal,
    })
    if (!resp.ok || !resp.body) {
      throw new KeystoneApiError(`Stream connection failed (${resp.status})`, resp.status)
    }

    const reader = resp.body.getReader()
    const decoder = new SseFrameDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      for (const ev of parsePayloads<TaskEvent>(decoder.push(value))) onEvent(ev)
    }
  }
}
