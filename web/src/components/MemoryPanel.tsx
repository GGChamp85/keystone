import { useEffect, useState } from 'react'
import {
  approveMemory,
  createMemory,
  forgetMemory,
  listMemories,
  pinMemory,
  unpinMemory,
  type MemoryRecord,
} from '../api'

const KIND_OPTIONS = ['convention', 'preference', 'fact', 'avoid']

export function MemoryPanel({ onBack }: { onBack: () => void }) {
  const [memories, setMemories] = useState<MemoryRecord[]>([])
  const [repositoryFilter, setRepositoryFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)

  const [newContent, setNewContent] = useState('')
  const [newKind, setNewKind] = useState('fact')
  const [newRepository, setNewRepository] = useState('')
  const [newPinned, setNewPinned] = useState(false)
  const [adding, setAdding] = useState(false)

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const records = await listMemories({
        repository: repositoryFilter || undefined,
        status: statusFilter || undefined,
      })
      setMemories(records)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repositoryFilter, statusFilter])

  async function withBusy(id: string, action: () => Promise<MemoryRecord>) {
    setBusyId(id)
    setError(null)
    try {
      const updated = await action()
      setMemories((prev) => prev.map((m) => (m.id === updated.id ? updated : m)))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusyId(null)
    }
  }

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault()
    if (!newContent.trim()) return
    setAdding(true)
    setError(null)
    try {
      await createMemory({
        content: newContent,
        kind: newKind,
        repository: newRepository || undefined,
        pinned: newPinned,
      })
      setNewContent('')
      setNewRepository('')
      setNewPinned(false)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setAdding(false)
    }
  }

  return (
    <div className="memory-panel">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <h2>Memory</h2>
      <p className="plan-summary">
        What Keystone Agents has learned and will recall into future tasks on this team/repo — pinned memories
        are always included, proposed ones need approval before they're ever used.
      </p>

      <form className="task-form memory-add-form" onSubmit={handleAdd}>
        <label htmlFor="mem-content">Add a memory</label>
        <textarea
          id="mem-content"
          rows={2}
          placeholder="e.g. Never touch the legacy billing module — it is being decommissioned."
          value={newContent}
          onChange={(e) => setNewContent(e.target.value)}
        />
        <div className="form-row">
          <div>
            <label htmlFor="mem-kind">Kind</label>
            <select id="mem-kind" value={newKind} onChange={(e) => setNewKind(e.target.value)}>
              {KIND_OPTIONS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="mem-repo">Repository (optional — blank means team-wide)</label>
            <input
              id="mem-repo"
              type="text"
              placeholder="https://your-git-server/org/repo"
              value={newRepository}
              onChange={(e) => setNewRepository(e.target.value)}
            />
          </div>
        </div>
        <label className="memory-pin-checkbox">
          <input type="checkbox" checked={newPinned} onChange={(e) => setNewPinned(e.target.checked)} />
          Pin (always recalled, regardless of task relevance)
        </label>
        <button type="submit" disabled={adding || !newContent.trim()}>
          {adding ? 'Adding…' : 'Add memory'}
        </button>
      </form>

      <div className="form-row memory-filters">
        <div>
          <label htmlFor="filter-repo">Filter by repository</label>
          <input
            id="filter-repo"
            type="text"
            placeholder="https://your-git-server/org/repo"
            value={repositoryFilter}
            onChange={(e) => setRepositoryFilter(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="filter-status">Filter by status</label>
          <select id="filter-status" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">all</option>
            <option value="approved">approved</option>
            <option value="proposed">proposed</option>
            <option value="forgotten">forgotten</option>
          </select>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}
      {loading && <p className="plan-summary">Loading…</p>}

      {!loading && memories.length === 0 && <p className="plan-summary">No memories match these filters.</p>}

      <ul className="memory-list">
        {memories.map((m) => (
          <li key={m.id} className={`memory-item memory-status-${m.status}`}>
            <div className="memory-item-header">
              <span className={`memory-badge memory-scope-${m.scope}`}>{m.scope}</span>
              <span className="memory-badge">{m.kind}</span>
              <span className={`memory-badge memory-status-badge-${m.status}`}>{m.status}</span>
              {m.pinned && <span className="memory-badge memory-pinned-badge">📌 pinned</span>}
              <span className="trace-time">{m.source === 'auto' ? 'proposed by agent' : `by ${m.created_by ?? 'user'}`}</span>
            </div>
            <p className="memory-content">{m.content}</p>
            {m.repository && <code>{m.repository}</code>}
            <div className="memory-actions">
              {m.status === 'proposed' && (
                <button disabled={busyId === m.id} onClick={() => withBusy(m.id, () => approveMemory(m.id))}>
                  Approve
                </button>
              )}
              {m.status !== 'forgotten' && (
                <button disabled={busyId === m.id} onClick={() => withBusy(m.id, () => forgetMemory(m.id))}>
                  Forget
                </button>
              )}
              {m.pinned ? (
                <button disabled={busyId === m.id} onClick={() => withBusy(m.id, () => unpinMemory(m.id))}>
                  Unpin
                </button>
              ) : (
                <button disabled={busyId === m.id} onClick={() => withBusy(m.id, () => pinMemory(m.id))}>
                  Pin
                </button>
              )}
              <span className="trace-time">{m.hit_count} recalls</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
