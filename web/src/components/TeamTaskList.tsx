import { useEffect, useState } from 'react'
import { listTasks, type AgentTaskSummary } from '../api'

const STATUS_LABELS: Record<string, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  timed_out: 'Timed out',
}

export function TeamTaskList({ onOpenTask }: { onOpenTask: (taskId: string) => void }) {
  const [tasks, setTasks] = useState<AgentTaskSummary[]>([])
  const [userFilter, setUserFilter] = useState('')
  const [repoFilter, setRepoFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const rows = await listTasks({
        user_id: userFilter || undefined,
        repository_url: repoFilter || undefined,
      })
      setTasks(rows)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userFilter, repoFilter])

  return (
    <div className="team-task-list">
      <h2>Team tasks</h2>
      <p className="plan-summary">
        Every task across the tenant — who's working on what, and its PR status. Filter to one person or
        repository below.
      </p>

      <div className="form-row memory-filters">
        <div>
          <label htmlFor="task-filter-user">Filter by user id</label>
          <input
            id="task-filter-user"
            type="text"
            placeholder="A user id from the admin API"
            value={userFilter}
            onChange={(e) => setUserFilter(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="task-filter-repo">Filter by repository</label>
          <input
            id="task-filter-repo"
            type="text"
            placeholder="https://your-git-server/org/repo"
            value={repoFilter}
            onChange={(e) => setRepoFilter(e.target.value)}
          />
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}
      {loading && <p className="plan-summary">Loading…</p>}
      {!loading && tasks.length === 0 && <p className="plan-summary">No tasks match these filters.</p>}

      <ul className="task-list">
        {tasks.map((t) => (
          <li key={t.id} className={`task-item task-status-${t.status}`}>
            <div className="task-item-header">
              <span className={`task-badge task-status-badge-${t.status}`}>
                {STATUS_LABELS[t.status] ?? t.status}
              </span>
              <span className="task-badge">{t.model_role}</span>
              {t.user_id && <span className="trace-time">by {t.user_id.slice(0, 8)}</span>}
              <span className="trace-time">{new Date(t.created_at).toLocaleString()}</span>
            </div>
            <p className="task-description">{t.task_description}</p>
            <div className="task-item-meta">
              {t.repository_url && <code>{t.repository_url}</code>}
              {t.branch_name && <code>{t.branch_name}</code>}
            </div>
            <div className="task-item-actions">
              {t.pr_url && (
                <a className="task-pr-link" href={t.pr_url} target="_blank" rel="noreferrer">
                  View PR{t.pr_number ? ` #${t.pr_number}` : ''}
                </a>
              )}
              <button className="back-link" onClick={() => onOpenTask(t.id)}>
                Open task →
              </button>
            </div>
            {t.error_message && <p className="error-text">{t.error_message}</p>}
          </li>
        ))}
      </ul>
    </div>
  )
}
