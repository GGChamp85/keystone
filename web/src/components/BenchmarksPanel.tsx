import { useEffect, useState } from 'react'
import { getBenchmarks, type BenchmarkComparison } from '../api'

function pct(v: number): string {
  return `${Math.round(v * 100)}%`
}

function seconds(ms: number): string {
  return ms ? `${(ms / 1000).toFixed(0)} s` : '—'
}

export function BenchmarksPanel({ onBack }: { onBack: () => void }) {
  const [days, setDays] = useState<number | null>(null)
  const [data, setData] = useState<BenchmarkComparison | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
    getBenchmarks(days)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [days])

  const labels = data?.backends.map((b) => b.label) ?? []

  return (
    <div className="benchmarks-panel">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <div className="model-library-head">
        <h2>Benchmarks</h2>
        <div className="spend-range">
          {[null, 7, 30].map((d) => (
            <button
              key={String(d)}
              className={`nav-link${d === days ? ' nav-link-active' : ''}`}
              onClick={() => setDays(d)}
            >
              {d === null ? 'all runs' : `${d} days`}
            </button>
          ))}
        </div>
      </div>
      <p className="steps-hint">
        Which backend solved which repository task, how fast, at what token cost — the latest persisted run of
        each task per backend (<code>benchmark_runs</code>, written by <code>benchmarks/agent_runner.py --persist</code>{' '}
        and <code>benchmarks/compare.py</code>). A solved task means its held-out tests really passed in a fresh
        clone of the pushed branch; nothing here is a model's opinion.
      </p>

      {error && <p className="error-text">{error}</p>}
      {data && data.backends.length === 0 && (
        <p className="steps-hint">
          No persisted runs yet. Run <code>python -m benchmarks.agent_runner --persist --backend-label &lt;name&gt;</code>{' '}
          against a backend, or <code>benchmarks/compare.py</code> across several, and this view fills in.
        </p>
      )}

      {data && data.backends.length > 0 && (
        <>
          <div className="stats-grid">
            {data.backends.map((b) => (
              <div className="stat" key={b.label}>
                <span className="stat-label">
                  {b.label}
                  {b.model_id ? ` · ${b.model_id}` : ''}
                </span>
                <span className="stat-value">
                  {b.solved}/{b.tasks_run} solved ({pct(b.solve_rate)})
                </span>
                <span className="stat-label">
                  avg {seconds(b.avg_duration_ms)} · {(b.total_prompt_tokens + b.total_completion_tokens).toLocaleString()} tokens
                </span>
              </div>
            ))}
          </div>

          <div className="spend-table-wrap">
            <table className="spend-table">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Language</th>
                  {labels.map((l) => (
                    <th key={l}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.tasks.map((t) => (
                  <tr key={t.task_id}>
                    <td>
                      <code>{t.task_id}</code>
                    </td>
                    <td>{t.language}</td>
                    {labels.map((l) => {
                      const r = t.results[l]
                      if (!r) return <td key={l}>—</td>
                      const cls = r.solved ? 'model-state model-state-ready' : 'model-state model-state-unhealthy'
                      return (
                        <td key={l}>
                          <span className={cls} title={r.error_message ?? r.status}>
                            {r.solved ? 'solved' : r.status}
                          </span>{' '}
                          <span className="steps-hint">
                            {seconds(r.duration_ms)} · {(r.prompt_tokens + r.completion_tokens).toLocaleString()} tok
                            {r.pr_url ? (
                              <>
                                {' · '}
                                <a href={r.pr_url} target="_blank" rel="noreferrer">
                                  PR
                                </a>
                              </>
                            ) : null}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="steps-hint">
            {data.runs_considered} run(s) considered; generated {new Date(data.generated_at * 1000).toLocaleString()}.
          </p>
        </>
      )}
    </div>
  )
}
