import { useEffect, useRef, useState } from 'react'
import {
  FINETUNE_TERMINAL_STATUSES,
  getFineTuneJob,
  listFineTuneJobs,
  previewFineTuneDataset,
  promoteFineTuneJob,
  rollbackFineTuneJob,
  startFineTuneJob,
  streamFineTuneJob,
  type DatasetPreview,
  type FineTuneEvent,
  type FineTuneJob,
} from '../api'

const JOB_TYPES = ['lora', 'sft', 'dpo']

type ConfigRow = { key: string; value: string }

function ConfigureStep({ onStarted }: { onStarted: (jobId: string) => void }) {
  const [jobType, setJobType] = useState('lora')
  const [baseModel, setBaseModel] = useState('Qwen/Qwen2.5-Coder-32B-Instruct')
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

  async function handlePromote() {
    setBusy(true)
    setActionError(null)
    try {
      const updated = await promoteFineTuneJob(jobId)
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
          <span className="stat-label">Latest message</span>
          <span className="stat-value">{latest?.message ?? '—'}</span>
        </div>
      </div>

      {isDone && (
        <div className={`result-card ${status === 'completed' ? 'result-ok' : 'result-error'}`}>
          <h3>{status === 'completed' ? 'Verdict' : 'Failure'}</h3>
          {status === 'completed' ? (
            <>
              <p>Training completed. Real metrics from this run:</p>
              <pre className="finetune-dataset-record">{JSON.stringify(job?.metrics ?? latest?.metrics ?? {}, null, 2)}</pre>
              {job?.output_model_path && <p>Adapter: <code>{job.output_model_path}</code></p>}
              <div className="finetune-verdict-actions">
                {canPromote && (
                  <button disabled={busy} onClick={handlePromote}>
                    {busy ? 'Promoting…' : 'Promote to default'}
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
  const [mode, setMode] = useState<'list' | 'configure' | 'progress'>('list')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  return (
    <div className="finetune-panel">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <h2>Fine-tune</h2>
      <p className="plan-summary">
        Train a LoRA/SFT/DPO adapter on your own repositories — pick a real training-data JSONL, watch progress
        live, then promote or roll back the result.
      </p>

      {mode === 'list' && (
        <JobListView
          onOpenJob={(id) => {
            setActiveJobId(id)
            setMode('progress')
          }}
          onNewJob={() => setMode('configure')}
        />
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
