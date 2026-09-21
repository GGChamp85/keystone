// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Talks to the Keystone Inference / Keystone Agents API on the same
// origin this SPA is served from (src/main.py mounts web/dist at /app/,
// the API at /v1/*) — no separate base URL to configure.

const API_KEY_STORAGE_KEY = 'keystone_api_key'

export function getApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE_KEY) || ''
}

export function setApiKey(key: string): void {
  localStorage.setItem(API_KEY_STORAGE_KEY, key)
}

function authHeaders(): HeadersInit {
  const key = getApiKey()
  return key ? { Authorization: `Bearer ${key}` } : {}
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

export async function submitTask(req: TaskSubmitRequest): Promise<TaskSubmitResponse> {
  const resp = await fetch('/v1/keystone/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    throw new Error(`Task submission failed (${resp.status}): ${await resp.text()}`)
  }
  return resp.json()
}

export interface ModelInfo {
  id: string
  root: string
}

export async function listModels(): Promise<ModelInfo[]> {
  const resp = await fetch('/v1/models', { headers: authHeaders() })
  if (!resp.ok) {
    throw new Error(`Failed to list models (${resp.status})`)
  }
  const body = await resp.json()
  return body.data as ModelInfo[]
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

export interface ListMemoriesParams {
  repository?: string
  scope?: string
  status?: string
}

export async function listMemories(params: ListMemoriesParams = {}): Promise<MemoryRecord[]> {
  const query = new URLSearchParams()
  if (params.repository) query.set('repository', params.repository)
  if (params.scope) query.set('scope', params.scope)
  if (params.status) query.set('status', params.status)
  const qs = query.toString()
  const resp = await fetch(`/v1/keystone/memory${qs ? `?${qs}` : ''}`, { headers: authHeaders() })
  if (!resp.ok) {
    throw new Error(`Failed to list memories (${resp.status})`)
  }
  return resp.json()
}

export interface CreateMemoryRequest {
  content: string
  kind: string
  repository?: string
  pinned?: boolean
}

export async function createMemory(req: CreateMemoryRequest): Promise<MemoryRecord> {
  const resp = await fetch('/v1/keystone/memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    throw new Error(`Failed to create memory (${resp.status}): ${await resp.text()}`)
  }
  return resp.json()
}

async function memoryAction(id: string, action: 'approve' | 'forget' | 'pin' | 'unpin'): Promise<MemoryRecord> {
  const resp = await fetch(`/v1/keystone/memory/${id}/${action}`, {
    method: 'POST',
    headers: authHeaders(),
  })
  if (!resp.ok) {
    throw new Error(`Failed to ${action} memory (${resp.status})`)
  }
  return resp.json()
}

export const approveMemory = (id: string): Promise<MemoryRecord> => memoryAction(id, 'approve')
export const forgetMemory = (id: string): Promise<MemoryRecord> => memoryAction(id, 'forget')
export const pinMemory = (id: string): Promise<MemoryRecord> => memoryAction(id, 'pin')
export const unpinMemory = (id: string): Promise<MemoryRecord> => memoryAction(id, 'unpin')

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

export async function listTasks(params: ListTasksParams = {}): Promise<AgentTaskSummary[]> {
  const query = new URLSearchParams()
  if (params.user_id) query.set('user_id', params.user_id)
  if (params.repository_url) query.set('repository_url', params.repository_url)
  if (params.limit) query.set('limit', String(params.limit))
  const qs = query.toString()
  const resp = await fetch(`/v1/keystone/tasks${qs ? `?${qs}` : ''}`, { headers: authHeaders() })
  if (!resp.ok) {
    throw new Error(`Failed to list tasks (${resp.status})`)
  }
  return resp.json()
}

// Mirrors src/orchestrator/events.py's _summarize() payload exactly —
// keep these two in sync.
export interface TaskEvent {
  node: string
  phase: string
  iteration: number
  max_iterations: number | null
  plan: string
  plan_steps: string[]
  current_plan_step: number
  files_changed: string[]
  review_passed: boolean | null
  review_comment_count: number
  tests_passed: boolean | null
  test_pass_count: number
  test_total_count: number
  total_tokens: number
  result_summary: string
  error_message: string | null
  timestamp: number
}

export const TERMINAL_PHASES = new Set(['complete', 'failed', 'cancelled'])

/**
 * Reads GET /v1/keystone/tasks/{id}/stream as Server-Sent Events using
 * `fetch` + a manual ReadableStream reader rather than the native
 * `EventSource` — EventSource cannot set an Authorization header, and
 * this project's whole API is Bearer-token authenticated; a fetch-based
 * reader keeps that one consistent auth path instead of adding a
 * query-string API key just for this one endpoint.
 */
export async function streamTask(
  taskId: string,
  onEvent: (ev: TaskEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch(`/v1/keystone/tasks/${taskId}/stream`, {
    headers: authHeaders(),
    signal,
  })
  if (!resp.ok || !resp.body) {
    throw new Error(`Stream connection failed (${resp.status})`)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const dataLine = frame.split('\n').find((line) => line.startsWith('data: '))
      if (!dataLine) continue // keep-alive comment lines (": keep-alive") have no "data: " prefix
      try {
        onEvent(JSON.parse(dataLine.slice('data: '.length)) as TaskEvent)
      } catch {
        // malformed frame — skip rather than crash the whole stream
      }
    }
  }
}
