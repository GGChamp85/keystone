import { useEffect, useRef, useState } from 'react'
import { streamTask, TERMINAL_PHASES, type TaskEvent } from '../api'

const PHASE_ORDER = ['planning', 'coding', 'review', 'testing', 'fixing', 'complete'] as const

const PHASE_LABELS: Record<string, string> = {
  planning: 'Planning',
  coding: 'Coding',
  review: 'Review',
  testing: 'Testing',
  fixing: 'Fixing',
  complete: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function PhaseTimeline({ current }: { current: string }) {
  const currentIndex = PHASE_ORDER.indexOf(current as (typeof PHASE_ORDER)[number])
  const failed = current === 'failed' || current === 'cancelled'

  return (
    <div className="phase-timeline">
      {PHASE_ORDER.map((phase, i) => {
        let state: 'done' | 'active' | 'pending' = 'pending'
        if (failed) {
          state = i < currentIndex ? 'done' : 'pending'
        } else if (currentIndex >= 0) {
          if (i < currentIndex) state = 'done'
          else if (i === currentIndex) state = 'active'
        }
        return (
          <div key={phase} className={`phase-step phase-${state}`}>
            <span className="phase-dot" />
            <span className="phase-label">{PHASE_LABELS[phase]}</span>
          </div>
        )
      })}
      {failed && <div className="phase-step phase-active phase-failed">{PHASE_LABELS[current]}</div>}
    </div>
  )
}

export function TaskStreamView({ taskId, onBack }: { taskId: string; onBack: () => void }) {
  const [events, setEvents] = useState<TaskEvent[]>([])
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [connected, setConnected] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    setEvents([])
    setConnectionError(null)
    setConnected(true)

    streamTask(
      taskId,
      (ev) => setEvents((prev) => [...prev, ev]),
      controller.signal,
    )
      .catch((err) => {
        if (controller.signal.aborted) return
        setConnectionError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => setConnected(false))

    return () => controller.abort()
  }, [taskId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [events])

  const latest = events[events.length - 1]
  const phase = latest?.phase ?? 'planning'
  const isDone = latest ? TERMINAL_PHASES.has(phase) : false

  return (
    <div className="task-stream-view">
      <button className="back-link" onClick={onBack}>
        ← New task
      </button>
      <h2>
        Task <code>{taskId}</code>
      </h2>

      <PhaseTimeline current={phase} />

      {latest?.plan_steps && latest.plan_steps.length > 0 && (
        <div className="plan-card">
          <h3>Plan</h3>
          {latest.plan && <p className="plan-summary">{latest.plan}</p>}
          <ol className="plan-steps">
            {latest.plan_steps.map((step, i) => (
              <li
                key={i}
                className={
                  i < latest.current_plan_step ? 'step-done' : i === latest.current_plan_step ? 'step-active' : ''
                }
              >
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="stats-grid">
        <div className="stat">
          <span className="stat-label">Iteration</span>
          <span className="stat-value">
            {latest?.iteration ?? 0}
            {latest?.max_iterations ? ` / ${latest.max_iterations}` : ''}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Files changed</span>
          <span className="stat-value">{latest?.files_changed.length ?? 0}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Review</span>
          <span className="stat-value">
            {latest?.review_passed === null || latest?.review_passed === undefined
              ? '—'
              : latest.review_passed
                ? 'Passed'
                : `${latest.review_comment_count} comment(s)`}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Tests</span>
          <span className="stat-value">
            {latest && latest.test_total_count > 0 ? `${latest.test_pass_count}/${latest.test_total_count} passed` : '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Tokens used</span>
          <span className="stat-value">{latest?.total_tokens?.toLocaleString() ?? 0}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Status</span>
          <span className="stat-value">{connected ? 'Live' : isDone ? 'Finished' : 'Disconnected'}</span>
        </div>
      </div>

      {latest?.files_changed && latest.files_changed.length > 0 && (
        <div className="files-card">
          <h3>Files touched</h3>
          <ul>
            {latest.files_changed.map((f) => (
              <li key={f}>
                <code>{f}</code>
              </li>
            ))}
          </ul>
        </div>
      )}

      {isDone && (
        <div className={`result-card ${phase === 'complete' ? 'result-ok' : 'result-error'}`}>
          <h3>{phase === 'complete' ? 'Result' : 'Failure'}</h3>
          <p>{latest.result_summary || latest.error_message || 'No summary available.'}</p>
        </div>
      )}

      {connectionError && <p className="error-text">Stream error: {connectionError}</p>}

      <details className="trace-log">
        <summary>Raw execution trace ({events.length} events)</summary>
        <div className="trace-scroll" ref={scrollRef}>
          {events.map((ev, i) => (
            <div key={i} className="trace-line">
              <span className="trace-node">{ev.node}</span>
              <span className="trace-phase">{ev.phase}</span>
              <span className="trace-time">{new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}
