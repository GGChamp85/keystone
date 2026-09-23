import { useEffect, useState } from 'react'
import { getModelLibrary, type LibraryModel, type ModelState } from '../api'

const STATE_LABEL: Record<ModelState, string> = {
  READY: 'Ready',
  UNHEALTHY: 'Unhealthy',
  NOT_DEPLOYED: 'Not deployed',
  TRAINING: 'Training',
}

export function StateBadge({ state }: { state: ModelState }) {
  return <span className={`model-state model-state-${state.toLowerCase()}`}>{STATE_LABEL[state]}</span>
}

function FeatureChips({ features }: { features: Record<string, boolean | null> }) {
  const label: Record<string, string> = {
    chat_completions: 'chat',
    anthropic_messages: 'messages API',
    streaming: 'streaming',
    tools: 'tools',
    json_schema: 'json schema',
    lora_adapters: 'LoRA',
  }
  return (
    <div className="feature-chips">
      {Object.entries(features).map(([k, v]) => (
        <span
          key={k}
          className={`feature-chip${v === true ? ' feature-chip-on' : v === false ? ' feature-chip-off' : ''}`}
          title={v === null ? 'unknown for this model (not in the catalog)' : v ? 'supported' : 'not supported'}
        >
          {label[k] ?? k}
          {v === null ? ' ?' : ''}
        </span>
      ))}
    </div>
  )
}

async function copy(text: string, setNote: (s: string | null) => void) {
  try {
    await navigator.clipboard.writeText(text)
    setNote(`Copied ${text}`)
  } catch {
    setNote(`Copy failed — path is ${text}`)
  }
  setTimeout(() => setNote(null), 2500)
}

function deploySnippet(m: LibraryModel): string {
  if (m.kind === 'catalog') {
    return [
      `# Serve ${m.hf_source} as the coding_fallback role (needs ~${m.deployment?.vram_serve_gb ?? '?'} GB VRAM):`,
      `helm upgrade keystone helm/keystone \\`,
      `  --set vllm.coding_fallback.model=${m.hf_source} \\`,
      `  --set vllm.coding_fallback.servedModelName=${m.name.toLowerCase()} \\`,
      `  --set vllm.coding_fallback.toolParser=${m.tool_parser ?? 'hermes'}`,
      ``,
      `# Or RunPod Serverless, scaled to zero when idle:`,
      `keystone deploy runpod-serverless --model ${m.hf_source}`,
      ``,
      `# Then, so /v1/models and the library name it: CODING_FALLBACK_MODEL_ID=${m.hf_source}`,
    ].join('\n')
  }
  if (m.kind === 'adapter') {
    return [
      `# Promoted adapters are served through the base model's role — nothing to deploy separately.`,
      `# Serving pods mount FINETUNING_OUTPUT_DIR and read lora_modules.json on restart;`,
      `# a promote also live-loads via vLLM's /v1/load_lora_adapter.`,
      `# Adapter path: ${m.path}`,
    ].join('\n')
  }
  return [
    `# Role "${m.id}" is served by ${m.endpoint}`,
    `# Change the model behind it (Helm): --set vllm.${m.id}.model=<hf id>`,
    `# Change the endpoint it points at (.env): VLLM_${m.id.toUpperCase()}_URL=<url>`,
  ].join('\n')
}

function Detail({ m, onClose, onPlayground, onFineTune }: { m: LibraryModel; onClose: () => void; onPlayground: (id: string) => void; onFineTune: (base: string) => void }) {
  const [note, setNote] = useState<string | null>(null)
  const [showDeploy, setShowDeploy] = useState(false)
  const rows: [string, string][] = [
    ['Kind', m.kind],
    ['Source', m.hf_source],
    ['Provider', m.provider],
    ['Endpoint', m.endpoint ?? '—'],
    ['Served as', m.served_path ?? '—'],
    ['Parameters', m.params_b != null ? `${m.params_b} B` : 'unknown'],
    ['Context', m.context != null ? `${m.context.toLocaleString()} tokens (${m.context_source ?? 'catalog'})` : 'unknown'],
    ['License', m.license ?? 'unknown'],
    ['MoE', m.moe == null ? 'unknown' : m.moe ? 'yes' : 'no'],
    ['Tool parser', m.tool_parser ?? 'none / unknown'],
  ]
  if (m.deployment) {
    for (const [k, v] of Object.entries(m.deployment)) if (v != null) rows.push([k.replace(/_/g, ' '), String(v)])
  }
  if (m.kind === 'adapter') {
    rows.push(['Adapter status', `${m.adapter_status}${m.is_default ? ' (default for its base model)' : ''}`])
    rows.push(['Rank', String(m.rank)])
    rows.push(['Path', m.path ?? '—'])
    const verdict = (m.metrics as { verdict?: { status?: string; reason?: string } } | undefined)?.verdict
    if (verdict) rows.push(['Verdict', `${verdict.status}: ${verdict.reason}`])
  }
  const health = m.health as { breaker?: string; last_error?: string | null; consecutive_failures?: number; probed_seconds_ago?: number | null } | undefined
  return (
    <aside className="model-detail result-card">
      <div className="model-detail-head">
        <h3>{m.name}</h3>
        <StateBadge state={m.state} />
        <button className="nav-link" onClick={onClose}>
          Close
        </button>
      </div>
      <p className="task-description">{m.purpose}</p>
      <div className="model-actions">
        {m.served_path && (
          <button className="nav-link" onClick={() => copy(m.served_path!, setNote)}>
            Copy model id
          </button>
        )}
        <button className="nav-link" disabled={m.state !== 'READY'} onClick={() => onPlayground(m.served_path ?? m.id)} title={m.state !== 'READY' ? 'only a READY model can be tried' : ''}>
          Try in Playground
        </button>
        {m.kind !== 'adapter' && (
          <button className="nav-link" onClick={() => onFineTune(m.hf_source)}>
            Fine-tune
          </button>
        )}
        <button className="nav-link" onClick={() => setShowDeploy((v) => !v)}>
          {showDeploy ? 'Hide deploy' : 'Deploy'}
        </button>
      </div>
      {note && <p className="steps-hint">{note}</p>}
      {showDeploy && <pre className="model-snippet">{deploySnippet(m)}</pre>}
      <FeatureChips features={m.api_features} />
      <table className="spend-table model-meta">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th>{k}</th>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {health && (
        <p className="steps-hint">
          Router view: breaker {health.breaker ?? 'closed'}, {health.consecutive_failures ?? 0} consecutive failures
          {health.probed_seconds_ago != null ? `, probed ${health.probed_seconds_ago}s ago` : ''}
          {health.last_error ? ` — last error: ${health.last_error}` : ''}
        </p>
      )}
      {m.adapters && m.adapters.length > 0 && (
        <>
          <h4>Adapters on this base model</h4>
          <ul className="model-adapter-list">
            {m.adapters.map((a) => (
              <li key={a.name}>
                <code>{a.name}</code> — {a.status}
                {a.is_default ? ', default' : ''} (r={a.rank}, {a.job_type})
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  )
}

export function ModelLibrary({ onBack, onPlayground, onFineTune }: { onBack: () => void; onPlayground: (modelId: string) => void; onFineTune: (baseModel: string) => void }) {
  const [models, setModels] = useState<LibraryModel[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [refreshedAt, setRefreshedAt] = useState<number | null>(null)

  function load() {
    setError(null)
    getModelLibrary()
      .then((lib) => {
        setModels(lib.models)
        setRefreshedAt(lib.generated_at)
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }
  useEffect(load, [])

  const groups: { title: string; kind: LibraryModel['kind']; hint: string }[] = [
    { title: 'Serving roles', kind: 'role', hint: 'What a request with this `model` hits right now — state from the router\'s own health registry.' },
    { title: 'Your adapters', kind: 'adapter', hint: 'Fine-tuned on your repositories. A promoted default is what the router serves this tenant.' },
    { title: 'Catalog', kind: 'catalog', hint: 'Deployable, fine-tunable open-weight coders with the VRAM they need.' },
  ]
  const current = models?.find((m) => m.kind + ':' + m.id === selected) ?? null

  return (
    <div className="model-library">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <div className="model-library-head">
        <h2>Model Library</h2>
        <button className="nav-link" onClick={load}>
          Refresh
        </button>
        {refreshedAt && <span className="steps-hint">as of {new Date(refreshedAt * 1000).toLocaleTimeString()}</span>}
      </div>
      {error && <p className="error-text">{error}</p>}
      {!models && !error && <p className="steps-hint">Probing endpoints…</p>}
      <div className={`model-library-body${current ? ' with-detail' : ''}`}>
        <div>
          {models &&
            groups.map((g) => {
              const items = models.filter((m) => m.kind === g.kind)
              if (items.length === 0) return null
              return (
                <section key={g.kind} className="model-group">
                  <h3>{g.title}</h3>
                  <p className="steps-hint">{g.hint}</p>
                  <div className="model-grid">
                    {items.map((m) => {
                      const key = m.kind + ':' + m.id
                      return (
                        <button key={key} className={`model-card${key === selected ? ' model-card-active' : ''}`} onClick={() => setSelected(key)}>
                          <div className="model-card-head">
                            <strong>{m.name}</strong>
                            <StateBadge state={m.state} />
                          </div>
                          <code className="model-card-id">{m.served_path ?? m.id}</code>
                          <p className="task-description">{m.purpose}</p>
                          <div className="model-card-meta">
                            {m.params_b != null && <span>{m.params_b}B</span>}
                            {m.context != null && <span>{Math.round(m.context / 1024)}K ctx</span>}
                            {m.license && <span>{m.license}</span>}
                            <span>{m.provider}</span>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </section>
              )
            })}
        </div>
        {current && <Detail m={current} onClose={() => setSelected(null)} onPlayground={onPlayground} onFineTune={onFineTune} />}
      </div>
    </div>
  )
}
