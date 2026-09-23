import { useEffect, useRef, useState } from 'react'
import { getApiKey, getModelLibrary, streamChat, type ChatStreamResult, type ChatTurn, type LibraryModel } from '../api'
import { StateBadge } from './ModelLibrary'

interface Exchange {
  user: string
  assistant: string
  result: ChatStreamResult | null
  error: string | null
}

function curlFor(model: string, system: string, turns: ChatTurn[], temperature: number, maxTokens: number): string {
  const messages: ChatTurn[] = system.trim() ? [{ role: 'system', content: system }, ...turns] : turns
  const body = JSON.stringify({ model, messages, temperature, max_tokens: maxTokens, stream: true }, null, 2)
  return `curl -N ${window.location.origin}/v1/chat/completions \\\n  -H "Authorization: Bearer ${getApiKey() ? 'ks-…' : '<api key>'}" \\\n  -H "Content-Type: application/json" \\\n  -d '${body.replace(/'/g, "'\\''")}'`
}

export function Playground({ initialModel, onBack, onLibrary }: { initialModel: string | null; onBack: () => void; onLibrary: () => void }) {
  const [models, setModels] = useState<LibraryModel[]>([])
  const [model, setModel] = useState(initialModel ?? 'coding')
  const [system, setSystem] = useState('')
  const [temperature, setTemperature] = useState(0.2)
  const [maxTokens, setMaxTokens] = useState(1024)
  const [input, setInput] = useState('')
  const [history, setHistory] = useState<Exchange[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showCurl, setShowCurl] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    getModelLibrary()
      .then((lib) => setModels(lib.models.filter((m) => m.served_path && m.kind !== 'catalog')))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  const selected = models.find((m) => m.served_path === model || m.id === model) ?? null
  const turns: ChatTurn[] = history.flatMap((h) => [
    { role: 'user' as const, content: h.user },
    ...(h.assistant ? [{ role: 'assistant' as const, content: h.assistant }] : []),
  ])

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setError(null)
    setBusy(true)
    const messages: ChatTurn[] = [...(system.trim() ? [{ role: 'system' as const, content: system }] : []), ...turns, { role: 'user', content: text }]
    const idx = history.length
    setHistory((h) => [...h, { user: text, assistant: '', result: null, error: null }])
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const result = await streamChat(
        { model, messages, temperature, max_tokens: maxTokens },
        (delta) => setHistory((h) => h.map((e, i) => (i === idx ? { ...e, assistant: e.assistant + delta } : e))),
        controller.signal,
      )
      setHistory((h) => h.map((e, i) => (i === idx ? { ...e, result } : e)))
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setHistory((h) => h.map((e, i) => (i === idx ? { ...e, error: message } : e)))
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  return (
    <div className="playground">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <div className="model-library-head">
        <h2>Playground</h2>
        <button className="nav-link" onClick={onLibrary}>
          Model Library
        </button>
      </div>
      <p className="steps-hint">
        A streamed request to <code>/v1/chat/completions</code> — the exact call any OpenAI client makes — with the
        backend's real token usage and timings. Nothing here is special-cased for the UI.
      </p>
      <div className="playground-controls">
        <label>
          Model
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {!models.some((m) => (m.served_path ?? m.id) === model) && <option value={model}>{model}</option>}
            {models.map((m) => (
              <option key={m.kind + m.id} value={m.served_path ?? m.id} disabled={m.state !== 'READY'}>
                {m.served_path ?? m.id} — {m.name} ({m.state.toLowerCase()})
              </option>
            ))}
          </select>
        </label>
        <label>
          Temperature
          <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
        </label>
        <label>
          Max tokens
          <input type="number" min={1} step={1} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
        </label>
        {selected && (
          <span className="playground-selected">
            <StateBadge state={selected.state} /> {selected.hf_source}
            {selected.context ? ` · ${Math.round(selected.context / 1024)}K context` : ''}
          </span>
        )}
      </div>
      <label className="playground-system">
        System prompt (optional)
        <textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} placeholder="You are a senior engineer on this codebase…" />
      </label>

      {error && <p className="error-text">{error}</p>}

      <div className="playground-transcript">
        {history.length === 0 && <p className="steps-hint">No messages yet.</p>}
        {history.map((h, i) => (
          <div key={i} className="playground-exchange">
            <div className="playground-msg playground-user">
              <span className="stat-label">you</span>
              <pre>{h.user}</pre>
            </div>
            <div className="playground-msg playground-assistant">
              <span className="stat-label">{model}</span>
              <pre>{h.assistant || (busy && i === history.length - 1 ? '…' : '')}</pre>
              {h.error && <p className="error-text">{h.error}</p>}
              {h.result && (
                <p className="playground-stats">
                  {h.result.usage
                    ? `${h.result.usage.prompt_tokens} prompt + ${h.result.usage.completion_tokens} completion tokens`
                    : 'usage not reported by the backend'}
                  {h.result.ttft_ms != null ? ` · first token ${Math.round(h.result.ttft_ms)} ms` : ''}
                  {` · total ${(h.result.elapsed_ms / 1000).toFixed(2)} s`}
                  {h.result.usage && h.result.elapsed_ms > 0
                    ? ` · ${(h.result.usage.completion_tokens / (h.result.elapsed_ms / 1000)).toFixed(1)} tok/s`
                    : ''}
                  {h.result.served_role ? ` · served by role ${h.result.served_role}` : ''}
                  {h.result.finish_reason ? ` · finish: ${h.result.finish_reason}` : ''}
                </p>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form
        className="playground-input"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <textarea
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the model something… (Enter to send, Shift+Enter for a new line)"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
        />
        <div className="playground-buttons">
          <button type="submit" disabled={busy || !input.trim()}>
            Send
          </button>
          {busy && (
            <button type="button" className="nav-link" onClick={stop}>
              Stop
            </button>
          )}
          <button type="button" className="nav-link" onClick={() => setHistory([])} disabled={busy || history.length === 0}>
            Clear
          </button>
          <button type="button" className="nav-link" onClick={() => setShowCurl((v) => !v)}>
            {showCurl ? 'Hide curl' : 'Show as curl'}
          </button>
        </div>
      </form>
      {showCurl && <pre className="model-snippet">{curlFor(model, system, [...turns, ...(input.trim() ? [{ role: 'user' as const, content: input }] : [])], temperature, maxTokens)}</pre>}
    </div>
  )
}
