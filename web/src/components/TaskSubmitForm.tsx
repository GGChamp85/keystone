import { useEffect, useState } from 'react'
import { listModels, submitTask, type ModelInfo } from '../api'

export function TaskSubmitForm({ onSubmitted }: { onSubmitted: (taskId: string) => void }) {
  const [task, setTask] = useState('')
  const [repositoryUrl, setRepositoryUrl] = useState('')
  const [branch, setBranch] = useState('main')
  const [model, setModel] = useState('coding')
  const [maxIterations, setMaxIterations] = useState(10)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listModels()
      .then(setModels)
      .catch(() => {
        // Fall back to the three fixed roles if /v1/models isn't reachable
        // yet (e.g. no API key set) — the <select> still works either way.
        setModels([
          { id: 'coding', root: 'coding' },
          { id: 'coding_fallback', root: 'coding_fallback' },
          { id: 'reasoning', root: 'reasoning' },
        ])
      })
  }, [])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!task.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      const resp = await submitTask({
        task,
        repository_url: repositoryUrl || undefined,
        branch,
        model,
        max_iterations: maxIterations,
      })
      onSubmitted(resp.task_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form className="task-form" onSubmit={handleSubmit}>
      <h2>New Keystone Agents task</h2>

      <label htmlFor="task">Task description</label>
      <textarea
        id="task"
        rows={4}
        placeholder="Add cursor-based pagination to the /users endpoint"
        value={task}
        onChange={(e) => setTask(e.target.value)}
        required
      />

      <div className="form-row">
        <div>
          <label htmlFor="repo">Repository URL (optional)</label>
          <input
            id="repo"
            type="text"
            placeholder="https://your-git-server/org/repo"
            value={repositoryUrl}
            onChange={(e) => setRepositoryUrl(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="branch">Branch</label>
          <input id="branch" type="text" value={branch} onChange={(e) => setBranch(e.target.value)} />
        </div>
      </div>

      <div className="form-row">
        <div>
          <label htmlFor="model">Model</label>
          <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id} ({m.root})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="max-iter">Max iterations</label>
          <input
            id="max-iter"
            type="number"
            min={1}
            max={50}
            value={maxIterations}
            onChange={(e) => setMaxIterations(Number(e.target.value))}
          />
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      <button type="submit" disabled={submitting}>
        {submitting ? 'Submitting…' : 'Submit task'}
      </button>
    </form>
  )
}
