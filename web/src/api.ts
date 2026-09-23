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

// ── Fine-tuning (src/api/routes/finetune.py) ────────────────────────

export interface FineTuneJob {
  id: string
  base_model: string
  job_type: string
  status: string
  config: Record<string, unknown>
  metrics: Record<string, unknown>
  output_model_path: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
  error_message: string | null
}

export interface StartFineTuneRequest {
  job_type: string
  base_model: string
  training_data_path: string
  config: Record<string, unknown>
}

export async function startFineTuneJob(req: StartFineTuneRequest): Promise<FineTuneJob> {
  const resp = await fetch('/v1/finetune/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    throw new Error(`Failed to start job (${resp.status}): ${await resp.text()}`)
  }
  return resp.json()
}

export async function listFineTuneJobs(status?: string): Promise<FineTuneJob[]> {
  const qs = status ? `?status=${encodeURIComponent(status)}` : ''
  const resp = await fetch(`/v1/finetune/jobs${qs}`, { headers: authHeaders() })
  if (!resp.ok) {
    throw new Error(`Failed to list jobs (${resp.status})`)
  }
  return resp.json()
}

export async function getFineTuneJob(jobId: string): Promise<FineTuneJob> {
  const resp = await fetch(`/v1/finetune/jobs/${jobId}`, { headers: authHeaders() })
  if (!resp.ok) {
    throw new Error(`Failed to fetch job (${resp.status})`)
  }
  return resp.json()
}

export interface DatasetPreview {
  path: string
  total_records: number
  preview: Record<string, unknown>[]
}

export async function previewFineTuneDataset(jobId: string, lines = 5): Promise<DatasetPreview> {
  const resp = await fetch(`/v1/finetune/jobs/${jobId}/dataset-preview?lines=${lines}`, {
    headers: authHeaders(),
  })
  if (!resp.ok) {
    throw new Error(`Failed to preview dataset (${resp.status}): ${await resp.text()}`)
  }
  return resp.json()
}

async function fineTuneJobAction(jobId: string, action: string): Promise<FineTuneJob> {
  const resp = await fetch(`/v1/finetune/jobs/${jobId}/${action}`, {
    method: 'POST',
    headers: authHeaders(),
  })
  if (!resp.ok) {
    throw new Error(`Failed to ${action} job (${resp.status}): ${await resp.text()}`)
  }
  return resp.json()
}

export const promoteFineTuneJob = (jobId: string, force = false): Promise<FineTuneJob> =>
  fineTuneJobAction(jobId, force ? 'promote?force=true' : 'promote')
export const rollbackFineTuneJob = (jobId: string): Promise<FineTuneJob> => fineTuneJobAction(jobId, 'rollback')

// Mirrors src/finetuning/events.py's publish_finetune_event() payload.
export interface FineTuneEvent {
  status: string
  message: string
  metrics: Record<string, unknown>
  timestamp: number
}

export const FINETUNE_TERMINAL_STATUSES = new Set(['completed', 'failed'])

export async function streamFineTuneJob(
  jobId: string,
  onEvent: (ev: FineTuneEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch(`/v1/finetune/jobs/${jobId}/stream`, {
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
      if (!dataLine) continue
      try {
        onEvent(JSON.parse(dataLine.slice('data: '.length)) as FineTuneEvent)
      } catch {
        // malformed frame — skip rather than crash the whole stream
      }
    }
  }
}


// ── Usage ledger (src/billing/ledger.py, GET /v1/keystone/usage) ─────────────

export interface UsageRow {
  day: string
  model_role: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  request_count: number
  estimated_cost_usd: number
}

export interface UsageSummary {
  tenant_id: string
  days: number
  since: string
  pricing_configured: boolean
  prices_per_million: Record<string, number>
  rows: UsageRow[]
  totals: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    request_count: number
    estimated_cost_usd: number
  }
}

export async function getUsage(days = 30): Promise<UsageSummary> {
  const resp = await fetch(`/v1/keystone/usage?days=${days}`, { headers: authHeaders() })
  if (!resp.ok) {
    throw new Error(`Failed to load usage (${resp.status})`)
  }
  return resp.json()
}


// ── Guided fine-tune (src/finetuning/guided.py; POST /v1/finetune/plans) ──────

export interface CatalogModel {
  hf_id: string
  short_name: string
  params_b: number
  license: string
  context: number
  vram_serve_gb: number
  vram_qlora_gb: number
  vram_lora_bf16_gb: number
  notes: string
}

export interface ModelCatalog {
  models: CatalogModel[]
  default: string
  detected_gpus: { name: string; vram_gb: number }[]
}

export interface GuidedPlanRequest {
  repositories: string[]
  goal: string
  base_model: string
  epochs: number
  holdout_ratio: number
  gpus?: { name: string; vram_gb: number }[]
  gpu_hourly_cost_usd?: number
}

export interface TrainingPlan {
  base_model: string
  method: string
  fits: boolean
  gpu_count: number
  gpu_name: string | null
  vram_available_gb: number
  vram_required_gb: number
  train_examples: number
  holdout_examples: number
  epochs: number
  total_steps: number
  lora_r: number
  estimated_tokens: number
  estimated_hours: number | null
  estimate_basis: string
  estimated_cost_usd: number | null
  reasons: string[]
}

export interface GuidedPlan {
  job_id: string
  status: string
  base_model: string
  dataset: {
    goal: string
    repositories: string[]
    per_repository: Record<string, number>
    trajectories: number
    total_examples: number
    dropped_by_secret_scan: number
    records_with_pii_redacted: number
    warnings: string[]
  }
  manifest: Record<string, unknown>
  plan: TrainingPlan
}

export async function getModelCatalog(): Promise<ModelCatalog> {
  const resp = await fetch('/v1/finetune/catalog', { headers: authHeaders() })
  if (!resp.ok) throw new Error(`Failed to load the model catalog (${resp.status})`)
  return resp.json()
}

export async function createGuidedPlan(req: GuidedPlanRequest): Promise<GuidedPlan> {
  const resp = await fetch('/v1/finetune/plans', {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!resp.ok) throw new Error(`Planning failed (${resp.status}): ${await resp.text()}`)
  return resp.json()
}

export async function approveGuidedPlan(jobId: string, force = false): Promise<FineTuneJob> {
  const resp = await fetch(`/v1/finetune/plans/${jobId}/approve`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ force }),
  })
  if (!resp.ok) throw new Error(`Approval failed (${resp.status}): ${await resp.text()}`)
  return resp.json()
}
