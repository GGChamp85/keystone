import { useEffect, useRef, useState } from 'react'
import {
  approveGuidedPlan,
  createGuidedPlan,
  FINETUNE_TERMINAL_STATUSES,
  getFineTuneJob,
  getModelCatalog,
  listFineTuneJobs,
  previewFineTuneDataset,
  promoteFineTuneJob,
  rollbackFineTuneJob,
  startFineTuneJob,
  streamFineTuneJob,
  type CatalogModel,
  type DatasetPreview,
  type FineTuneEvent,
  type FineTuneJob,
  type GuidedPlan,
} from '../api'

const JOB_TYPES = ['lora', 'sft', 'dpo']

type ConfigRow = { key: string; value: string }

function LossSparkline({ values }: { values: number[] }) {
  const w = 600
  const h = 120
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const points = values
    .map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${h - ((v - min) / span) * (h - 10) - 5}`)
    .join(' ')
  return (
    <svg className="loss-sparkline" viewBox={`0 0 ${w} ${h}`} width="100%" height={h} role="img" aria-label="training loss">
      <polyline fill="none" stroke="currentColor" strokeWidth="2" points={points} />
      <text x="4" y="12" fontSize="11">
        max {max.toFixed(3)}
      </text>
      <text x="4" y={h - 4} fontSize="11">
        min {min.toFixed(3)}
      </text>
    </svg>
  )
}

function GuidedStep({ onStarted }: { onStarted: (jobId: string) => void }) {
  const [catalog, setCatalog] = useState<CatalogModel[]>([])
  const [detected, setDetected] = useState<{ name: string; vram_gb: number }[]>([])
  const [repos, setRepos] = useState('')
  const [goal, setGoal] = useState('')
  const [baseModel, setBaseModel] = useState(() => new URLSearchParams(window.location.search).get('base_model') ?? 'auto')
  const [epochs, setEpochs] = useState(1)
  const [gpuPrice, setGpuPrice] = useState('')
  const [gpuSpec, setGpuSpec] = useState('')
  const [plan, setPlan] = useState<GuidedPlan | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getModelCatalog()
      .then((c) => {
        setCatalog(c.models)
        setDetected(c.detected_gpus)
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function handlePlan(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const gpus = gpuSpec.trim()
        ? gpuSpec.split(',').map((spec) => {
            const [name, vram] = spec.trim().split(':')
            return { name: name.trim(), vram_gb: Number(vram) }
          })
        : undefined
      const created = await createGuidedPlan({
        repositories: repos
          .split(/\s+/)
          .map((r) => r.trim())
          .filter(Boolean),
        goal: goal.trim(),
        base_model: baseModel,
        epochs,
        holdout_ratio: 0.2,
        gpus,
        gpu_hourly_cost_usd: gpuPrice.trim() ? Number(gpuPrice) : undefined,
      })
      setPlan(created)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function handleApprove(force: boolean) {
    if (!plan) return
    setBusy(true)
    setError(null)
    try {
      const job = await approveGuidedPlan(plan.job_id, force)
      onStarted(job.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  if (plan) {
    const p = plan.plan
    const ds = plan.dataset
    return (
      <div className="plan-card">
        <h3>Step 2 of 3 — Review the plan</h3>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Model</span>
            <span className="stat-value">{plan.base_model.split('/').pop()}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Method</span>
            <span className="stat-value">{p.method.toUpperCase()}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Fits</span>
            <span className="stat-value">{p.fits ? 'Yes' : 'No'}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Examples</span>
            <span className="stat-value">
              {p.train_examples} / {p.holdout_examples} held out
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Steps</span>
            <span className="stat-value">{p.total_steps}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Time</span>
            <span className="stat-value">{p.estimated_hours !== null ? `~${p.estimated_hours} h` : '—'}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Cost</span>
            <span className="stat-value">{p.estimated_cost_usd !== null ? `~$${p.estimated_cost_usd}` : '—'}</span>
          </div>
        </div>
        <p className="steps-hint">
          Hardware: {p.gpu_count} × {p.gpu_name ?? '?'} ({p.vram_available_gb} GB; needs ~{p.vram_required_gb} GB).
          {' '}
          {p.estimate_basis}
        </p>
        <p className="steps-hint">
          Sources:{' '}
          {Object.entries(ds.per_repository)
            .map(([r, n]) => `${r} (${n})`)
            .join(', ')}
          ; accepted agent tasks: {ds.trajectories}. Safety: {ds.dropped_by_secret_scan} record(s) dropped by the
          secret scan, {ds.records_with_pii_redacted} with PII redacted.
        </p>
        <ul className="plan-steps">
          {p.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
          {ds.warnings.map((w) => (
            <li key={w} className="error-text">
              {w}
            </li>
          ))}
        </ul>
        {error && <p className="error-text">{error}</p>}
        <div className="task-item-actions">
          <button disabled={busy || !p.fits} onClick={() => handleApprove(false)}>
            {busy ? 'Starting…' : 'Approve and train'}
          </button>
          {!p.fits && (
            <button disabled={busy} onClick={() => handleApprove(true)}>
              Start anyway (hardware not visible to the planner)
            </button>
          )}
          <button className="back-link" disabled={busy} onClick={() => setPlan(null)}>
            ← Change the description
          </button>
        </div>
      </div>
    )
  }

  return (
    <form className="task-form finetune-configure-form" onSubmit={handlePlan}>
      <h3>Step 1 of 3 — Describe</h3>
      <label htmlFor="gf-repos">Repositories to learn from (allow-listed git URLs, one per line)</label>
      <textarea id="gf-repos" rows={3} value={repos} onChange={(e) => setRepos(e.target.value)} />
      <label htmlFor="gf-goal">What should the adapter get better at?</label>
      <textarea
        id="gf-goal"
        rows={2}
        placeholder="Follow this codebase's conventions when fixing bugs in the payments service"
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
      />
      <label htmlFor="gf-model">Base model</label>
      <select id="gf-model" value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
        <option value="auto">auto — the default SLM, stepped down if it would not fit</option>
        {catalog.map((m) => (
          <option key={m.hf_id} value={m.hf_id}>
            {m.short_name} — {m.params_b}B, {m.license}, QLoRA ~{m.vram_qlora_gb} GB
          </option>
        ))}
      </select>
      <label htmlFor="gf-epochs">Epochs</label>
      <input id="gf-epochs" type="number" min={1} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
      <label htmlFor="gf-gpu">
        Target GPUs (NAME:VRAM_GB, comma-separated) — leave empty to use what the server detects
        {detected.length > 0 ? `: ${detected.map((g) => `${g.name} ${g.vram_gb} GB`).join(', ')}` : ' (none detected)'}
      </label>
      <input id="gf-gpu" type="text" placeholder="NVIDIA L4:22.5" value={gpuSpec} onChange={(e) => setGpuSpec(e.target.value)} />
      <label htmlFor="gf-price">Your GPU price (USD per GPU-hour, optional — for the cost line)</label>
      <input id="gf-price" type="text" placeholder="0.69" value={gpuPrice} onChange={(e) => setGpuPrice(e.target.value)} />
      {error && <p className="error-text">{error}</p>}
      <button type="submit" disabled={busy || !repos.trim() || goal.trim().length < 10}>
        {busy ? 'Building the dataset…' : 'Build the plan'}
      </button>
    </form>
  )
}

function ConfigureStep({ onStarted }: { onStarted: (jobId: string) => void }) {
  const [jobType, setJobType] = useState('lora')
  const [baseModel, setBaseModel] = useState('Qwen/Qwen2.5-Coder-7B-Instruct')
  const [trainingDataPath, setTrainingDataPath] = useState('')
  const [configRows, setConfigRows] = useState<ConfigRow[]>([{ key: 'num_epochs', value: '3' }])
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function updateRow(i: number, field: keyof ConfigRow, value: string) {
    setConfigRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, [field]: value } : r)))
  }

  function addRow() {
    setConfigRows((prev) => [...prev, { key: '', value: '' }])
  }

  function removeRow(i: number) {
    setConfigRows((prev) => prev.filter((_, idx) => idx !== i))
  }

  function buildConfig(): Record<string, unknown> {
    const config: Record<string, unknown> = {}
    for (const row of configRows) {
      if (!row.key.trim()) continue
      try {
        config[row.key.trim()] = JSON.parse(row.value)
      } catch {
        config[row.key.trim()] = row.value
      }
    }
    return config
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!trainingDataPath.trim()) return
    setStarting(true)
    setError(null)
    try {
      const job = await startFineTuneJob({
        job_type: jobType,
        base_model: baseModel,
        training_data_path: trainingDataPath.trim(),
        config: buildConfig(),
      })
      onStarted(job.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setStarting(false)
    }
  }

  return (
    <form className="task-form finetune-configure-form" onSubmit={handleSubmit}>
      <label htmlFor="ft-job-type">Job type</label>
      <select id="ft-job-type" value={jobType} onChange={(e) => setJobType(e.target.value)}>
        {JOB_TYPES.map((t) => (
          <option key={t} value={t}>
            {t.toUpperCase()}
          </option>
        ))}
      </select>

      <label htmlFor="ft-base-model">Base model</label>
      <input id="ft-base-model" type="text" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />

      <label htmlFor="ft-data-path">Training data path (JSONL, reachable by the server)</label>
      <input
        id="ft-data-path"
        type="text"
        placeholder="/data/finetune/my-repo-trajectories/train.jsonl"
        value={trainingDataPath}
        onChange={(e) => setTrainingDataPath(e.target.value)}
      />

      <div className="finetune-config-rows">
        <label>Advanced config overrides (optional)</label>
        {configRows.map((row, i) => (
          <div className="form-row finetune-config-row" key={i}>
            <input
              type="text"
              placeholder="key, e.g. lora_r"
              value={row.key}
              onChange={(e) => updateRow(i, 'key', e.target.value)}
            />
            <input
              type="text"
              placeholder="value, e.g. 16"
              value={row.value}
              onChange={(e) => updateRow(i, 'value', e.target.value)}
            />
            <button type="button" className="finetune-remove-row" onClick={() => removeRow(i)}>
              ✕
            </button>
          </div>
        ))}
        <button type="button" onClick={addRow} className="finetune-add-row">
          + Add override
        </button>
      </div>

      {error && <p className="error-text">{error}</p>}

      <button type="submit" disabled={starting || !trainingDataPath.trim()}>
        {starting ? 'Starting…' : 'Start fine-tuning job'}
      </button>
    </form>
  )
}

function JobProgressView({ jobId, onBack }: { jobId: string; onBack: () => void }) {
  const [job, setJob] = useState<FineTuneJob | null>(null)
  const [events, setEvents] = useState<FineTuneEvent[]>([])
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    getFineTuneJob(jobId).then(setJob).catch(() => undefined)
    previewFineTuneDataset(jobId).then(setPreview).catch((err) => setPreviewError(err.message))
  }, [jobId])

  useEffect(() => {
    const controller = new AbortController()
    setEvents([])
    setStreamError(null)
    streamFineTuneJob(jobId, (ev) => setEvents((prev) => [...prev, ev]), controller.signal).catch((err) => {
      if (controller.signal.aborted) return
      setStreamError(err instanceof Error ? err.message : String(err))
    })
    return () => controller.abort()
  }, [jobId])

  useEffect(() => {
    const latest = events[events.length - 1]
    if (latest && FINETUNE_TERMINAL_STATUSES.has(latest.status)) {
      getFineTuneJob(jobId).then(setJob).catch(() => undefined)
    }
  }, [events, jobId])

  const latest = events[events.length - 1]
  const status = latest?.status ?? job?.status ?? 'pending'
  const isDone = FINETUNE_TERMINAL_STATUSES.has(status)
  const canPromote = job?.status === 'completed' && !!job.output_model_path
  type Progress = { step: number; total_steps: number; loss?: number | null; eta_seconds?: number | null }
  const progress = events
    .map((ev) => ev.metrics as Partial<Progress> | undefined)
    .filter((m): m is Progress => !!m && typeof m.step === 'number' && typeof m.total_steps === 'number')
  const losses = progress.map((p) => p.loss).filter((l): l is number => typeof l === 'number')
  const lastLoss = losses.length ? losses[losses.length - 1] : null
  const eta = progress.length ? (progress[progress.length - 1].eta_seconds ?? null) : null
  const verdict = (job?.metrics?.verdict ?? null) as
    | { status: string; reason: string; base_eval_loss: number | null; eval_loss: number | null }
    | null

  async function handlePromote(force = false) {
    setBusy(true)
    setActionError(null)
    try {
      const updated = await promoteFineTuneJob(jobId, force)
      setJob(updated)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function handleRollback() {
    setBusy(true)
    setActionError(null)
    try {
      const updated = await rollbackFineTuneJob(jobId)
      setJob(updated)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="finetune-progress-view">
      <button className="back-link" onClick={onBack}>
        ← All jobs
      </button>
      <h3>
        {job?.job_type.toUpperCase() ?? '…'} on <code>{job?.base_model ?? jobId}</code>
      </h3>

      <div className="phase-timeline finetune-status-timeline">
        {['pending', 'running', 'completed'].map((s) => {
          const failed = status === 'failed'
          const order = ['pending', 'running', 'completed']
          const idx = order.indexOf(status === 'failed' ? 'running' : status)
          const i = order.indexOf(s)
          let state: 'done' | 'active' | 'pending' = 'pending'
          if (i < idx) state = 'done'
          else if (i === idx && !failed) state = 'active'
          return (
            <div key={s} className={`phase-step phase-${state}`}>
              <span className="phase-dot" />
              <span className="phase-label">{s}</span>
            </div>
          )
        })}
        {status === 'failed' && <div className="phase-step phase-active phase-failed">failed</div>}
      </div>

      {preview && (
        <div className="plan-card">
          <h3>Dataset preview</h3>
          <p className="plan-summary">
            <code>{preview.path}</code> — {preview.total_records} records
          </p>
          <div className="finetune-dataset-preview">
            {preview.preview.map((rec, i) => (
              <pre key={i} className="finetune-dataset-record">
                {JSON.stringify(rec, null, 2)}
              </pre>
            ))}
          </div>
        </div>
      )}
      {previewError && <p className="plan-summary">Dataset preview unavailable: {previewError}</p>}

      <div className="stats-grid">
        <div className="stat">
          <span className="stat-label">Status</span>
          <span className="stat-value">{status}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Step</span>
          <span className="stat-value">
            {progress.length > 0 ? `${progress[progress.length - 1].step} / ${progress[progress.length - 1].total_steps}` : '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Loss</span>
          <span className="stat-value">{lastLoss !== null ? lastLoss.toFixed(4) : '—'}</span>
        </div>
        <div className="stat">
          <span className="stat-label">ETA</span>
          <span className="stat-value">{eta !== null ? `${Math.round(eta / 60)} min` : '—'}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Latest message</span>
          <span className="stat-value">{latest?.message ?? '—'}</span>
        </div>
      </div>

      {losses.length > 1 && (
        <div className="plan-card">
          <h3>Training loss</h3>
          <LossSparkline values={losses} />
        </div>
      )}

      {isDone && (
        <div className={`result-card ${status === 'completed' ? 'result-ok' : 'result-error'}`}>
          <h3>{status === 'completed' ? 'Verdict' : 'Failure'}</h3>
          {status === 'completed' ? (
            <>
              {verdict && (
                <p>
                  <strong>
                    {verdict.status === 'pass'
                      ? '✅ Beats the base model'
                      : verdict.status === 'fail'
                        ? '❌ Does not beat the base model'
                        : '⚠️ No held-out comparison'}
                  </strong>
                  {' — '}
                  {verdict.reason}
                </p>
              )}
              <p>Real metrics from this run:</p>
              <pre className="finetune-dataset-record">{JSON.stringify(job?.metrics ?? latest?.metrics ?? {}, null, 2)}</pre>
              {job?.output_model_path && <p>Adapter: <code>{job.output_model_path}</code></p>}
              <div className="finetune-verdict-actions">
                {canPromote && (!verdict || verdict.status === 'pass') && (
                  <button disabled={busy} onClick={() => handlePromote(false)}>
                    {busy ? 'Promoting…' : 'Promote to default'}
                  </button>
                )}
                {canPromote && verdict && verdict.status !== 'pass' && (
                  <button disabled={busy} onClick={() => handlePromote(true)}>
                    {busy ? 'Promoting…' : 'Promote anyway (admin, audited)'}
                  </button>
                )}
                <button disabled={busy} onClick={handleRollback}>
                  {busy ? 'Rolling back…' : 'Rollback (retire adapter)'}
                </button>
              </div>
              {actionError && <p className="error-text">{actionError}</p>}
            </>
          ) : (
            <p>{job?.error_message || latest?.message || 'Training failed.'}</p>
          )}
        </div>
      )}

      {streamError && <p className="error-text">Stream error: {streamError}</p>}

      <details className="trace-log">
        <summary>Progress log ({events.length} events)</summary>
        <div className="trace-scroll">
          {events.map((ev, i) => (
            <div key={i} className="trace-line">
              <span className="trace-node">{ev.status}</span>
              <span className="trace-phase">{ev.message}</span>
              <span className="trace-time">{new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}

function JobListView({ onOpenJob, onNewJob }: { onOpenJob: (jobId: string) => void; onNewJob: () => void }) {
  const [jobs, setJobs] = useState<FineTuneJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  async function refresh() {
    try {
      const list = await listFineTuneJobs()
      setJobs(list)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    pollRef.current = window.setInterval(refresh, 10_000)
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current)
    }
  }, [])

  return (
    <div>
      <div className="finetune-list-header">
        <h3>Fine-tuning jobs</h3>
        <button onClick={onNewJob}>+ New job</button>
      </div>
      {error && <p className="error-text">{error}</p>}
      {loading && <p className="plan-summary">Loading…</p>}
      {!loading && jobs.length === 0 && (
        <p className="plan-summary">No fine-tuning jobs yet — start one to beat plain RAG on your own code.</p>
      )}
      <ul className="task-list">
        {jobs.map((j) => (
          <li key={j.id} className="task-item">
            <div className="task-item-header">
              <span className={`task-badge task-status-badge-${j.status}`}>{j.status}</span>
              <span className="task-badge">{j.job_type}</span>
              <span className="trace-time">{new Date(j.created_at).toLocaleString()}</span>
            </div>
            <p className="task-description">{j.base_model}</p>
            <div className="task-item-meta">
              <code>{j.id.slice(0, 8)}</code>
            </div>
            <div className="task-item-actions">
              <button className="back-link" onClick={() => onOpenJob(j.id)}>
                Open job →
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function FineTunePanel({ onBack }: { onBack: () => void }) {
  const [mode, setMode] = useState<'list' | 'guided' | 'configure' | 'progress'>('list')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  return (
    <div className="finetune-panel">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <h2>Fine-tune</h2>
      <p className="plan-summary">
        Describe what the adapter should learn, review the plan and its cost, approve, watch it train, then promote
        it — or bring your own JSONL for full control.
      </p>

      {mode === 'list' && (
        <JobListView
          onOpenJob={(id) => {
            setActiveJobId(id)
            setMode('progress')
          }}
          onNewJob={() => setMode('guided')}
        />
      )}

      {mode === 'guided' && (
        <>
          <button className="back-link" onClick={() => setMode('list')}>
            ← All jobs
          </button>
          <GuidedStep
            onStarted={(id) => {
              setActiveJobId(id)
              setMode('progress')
            }}
          />
          <p className="steps-hint">
            Prefer to supply your own training JSONL and every hyperparameter?{' '}
            <button className="back-link" onClick={() => setMode('configure')}>
              Advanced job →
            </button>
          </p>
        </>
      )}

      {mode === 'configure' && (
        <>
          <button className="back-link" onClick={() => setMode('list')}>
            ← All jobs
          </button>
          <ConfigureStep
            onStarted={(id) => {
              setActiveJobId(id)
              setMode('progress')
            }}
          />
        </>
      )}

      {mode === 'progress' && activeJobId && <JobProgressView jobId={activeJobId} onBack={() => setMode('list')} />}
    </div>
  )
}
