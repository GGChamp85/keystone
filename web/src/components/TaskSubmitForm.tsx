import { useEffect, useState } from 'react'
import { estimateTaskCost, listModels, submitTask, type ImageAttachment, type ModelInfo, type TaskCostEstimate } from '../api'

function usd(v: number): string {
  return `$${v.toFixed(v < 1 ? 4 : 2)}`
}

const ALLOWED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif'])
const MAX_IMAGE_BYTES = 10 * 1024 * 1024 // matches the backend's ~11 MB base64-decoded cap
const MAX_IMAGES = 4

function readImageAsAttachment(file: File): Promise<ImageAttachment> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      const data = result.slice(result.indexOf(',') + 1) // strip the "data:image/png;base64," prefix
      resolve({ media_type: file.type, data })
    }
    reader.onerror = () => reject(reader.error ?? new Error(`Failed to read ${file.name}`))
    reader.readAsDataURL(file)
  })
}

export function TaskSubmitForm({ onSubmitted }: { onSubmitted: (taskId: string) => void }) {
  const [task, setTask] = useState('')
  const [repositoryUrl, setRepositoryUrl] = useState('')
  const [branch, setBranch] = useState('main')
  const [model, setModel] = useState('coding')
  const [maxIterations, setMaxIterations] = useState(10)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [estimate, setEstimate] = useState<TaskCostEstimate | null>(null)
  const [estimating, setEstimating] = useState(false)
  const [images, setImages] = useState<{ name: string; attachment: ImageAttachment }[]>([])
  const [imageError, setImageError] = useState<string | null>(null)

  useEffect(() => {
    if (task.trim().length < 10) {
      setEstimate(null)
      return
    }
    const handle = setTimeout(() => {
      setEstimating(true)
      estimateTaskCost({ task, repository_url: repositoryUrl || undefined, model })
        .then(setEstimate)
        .catch(() => setEstimate(null))
        .finally(() => setEstimating(false))
    }, 600) // debounced: an estimate on every keystroke would just be noise
    return () => clearTimeout(handle)
  }, [task, repositoryUrl, model])

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

  async function handleImagesSelected(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return
    setImageError(null)
    const files = Array.from(fileList)
    if (images.length + files.length > MAX_IMAGES) {
      setImageError(`Up to ${MAX_IMAGES} images per task.`)
      return
    }
    for (const file of files) {
      if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
        setImageError(`${file.name}: unsupported type ${file.type || '(unknown)'} — PNG, JPEG, WebP, or GIF only.`)
        return
      }
      if (file.size > MAX_IMAGE_BYTES) {
        setImageError(`${file.name}: ${Math.round(file.size / 1024 / 1024)} MB exceeds the ${MAX_IMAGE_BYTES / 1024 / 1024} MB limit.`)
        return
      }
    }
    try {
      const attachments = await Promise.all(files.map(async (f) => ({ name: f.name, attachment: await readImageAsAttachment(f) })))
      setImages((prev) => [...prev, ...attachments])
    } catch (err) {
      setImageError(err instanceof Error ? err.message : String(err))
    }
  }

  function removeImage(index: number) {
    setImages((prev) => prev.filter((_, i) => i !== index))
  }

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
        images: images.length > 0 ? images.map((i) => i.attachment) : undefined,
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
      <p className="steps-hint">
        Describe the change in plain English. The agent clones the repository, plans, edits real files with
        tool calls, runs the repo's own tests, and opens a pull request when it's done — watch it happen live
        from the Tasks list after you submit. A repository URL is required for real work; branch defaults to{' '}
        <code>main</code>, and max iterations caps how many plan/fix cycles it gets before stopping.
      </p>

      <label htmlFor="task">Task description</label>
      <textarea
        id="task"
        rows={4}
        placeholder="Add cursor-based pagination to the /users endpoint"
        value={task}
        onChange={(e) => setTask(e.target.value)}
        required
      />

      <label htmlFor="images">
        Attach images (optional) — a bug screenshot, a UI mockup, a diagram
        <input
          id="images"
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          multiple
          onChange={(e) => {
            void handleImagesSelected(e.target.files)
            e.target.value = ''
          }}
          disabled={images.length >= MAX_IMAGES}
        />
      </label>
      <p className="steps-hint">
        Only used if the chosen model role is served by a vision-capable model — the task is refused otherwise
        rather than silently ignoring the image.
      </p>
      {imageError && <p className="error-text">{imageError}</p>}
      {images.length > 0 && (
        <ul className="attached-images">
          {images.map((img, i) => (
            <li key={`${img.name}-${i}`}>
              <img src={`data:${img.attachment.media_type};base64,${img.attachment.data}`} alt={img.name} />
              <span>{img.name}</span>
              <button type="button" className="nav-link" onClick={() => removeImage(i)}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="form-row">
        <div>
          <label htmlFor="repo">Repository URL</label>
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

      {estimating && <p className="steps-hint">Estimating cost…</p>}
      {estimate && (
        <p className="steps-hint">
          Estimate: ~{estimate.estimated_total_tokens.toLocaleString()} tokens
          {estimate.pricing_configured && estimate.estimated_cost_usd !== null
            ? ` (~${usd(estimate.estimated_cost_usd)})`
            : ' (no price configured for this role)'}
          {' — '}
          {estimate.estimate_basis}
        </p>
      )}

      {error && <p className="error-text">{error}</p>}

      <button type="submit" disabled={submitting}>
        {submitting ? 'Submitting…' : 'Submit task'}
      </button>
    </form>
  )
}
