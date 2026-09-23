import { useEffect, useRef, useState } from 'react'
import { isNodeEvent, isStepEvent, streamTask, TERMINAL_PHASES, type StepEvent, type TaskEvent } from '../api'

const PHASE_ORDER = ['planning', 'coding', 'quality', 'review', 'testing', 'fixing', 'complete'] as const

const PHASE_LABELS: Record<string, string> = {
  planning: 'Planning',
  coding: 'Coding',
  quality: 'Quality gates',
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

function str(v: unknown): string {
  if (v === null || v === undefined) return ''
  return typeof v === 'string' ? v : JSON.stringify(v, null, 2)
}

function describeStep(ev: StepEvent): { title: string; body: string; ok: boolean | null } {
  switch (ev.event_type) {
    case 'tool_call':
      return { title: `${str(ev.name)}(${str(ev.arguments)})`, body: str(ev.error), ok: ev.error ? false : null }
    case 'tool_result':
      return {
        title: `${str(ev.name)} → ${ev.ok ? 'ok' : 'error'}`,
        body: ev.ok ? str(ev.output) : str(ev.error),
        ok: Boolean(ev.ok),
      }
    case 'model_text':
      return { title: ev.final ? 'Model: final message' : 'Model', body: str(ev.content), ok: null }
    case 'test_output':
      return {
        title: `${str(ev.name)} → exit ${str(ev.exit_code)} (${ev.passed ? 'passed' : 'failed'})`,
        body: [str(ev.stdout), str(ev.stderr)].filter(Boolean).join('\n--- stderr ---\n'),
        ok: Boolean(ev.passed),
      }
    case 'quality_findings': {
      const findings = (ev.findings as unknown[]) ?? []
      const blocking = (ev.blocking as unknown[]) ?? []
      return {
        title: `${findings.length} finding(s), ${blocking.length} blocking`,
        body: str(findings),
        ok: blocking.length === 0,
      }
    }
    case 'deps_install':
      return { title: `${str(ev.command)} → ${ev.ok ? 'ok' : 'failed'}`, body: str(ev.output), ok: Boolean(ev.ok) }
    case 'repo_map':
      return { title: 'Repository map given to the planner and coder', body: str(ev.map), ok: null }
    case 'route_decision':
      return { title: `model=auto → ${str(ev.resolved_role)}`, body: str(ev), ok: null }
    case 'diff':
      return {
        title: `Committed ${str(ev.commit_sha).slice(0, 12)} on ${str(ev.branch_name)}`,
        body: str(ev.diff),
        ok: true,
      }
    case 'pr':
      return { title: `Pull request #${str(ev.pr_number)}: ${str(ev.pr_url)}`, body: '', ok: true }
    case 'candidate': {
      if (ev.winner !== null && ev.winner !== undefined) {
        return { title: `Best of ${str(ev.of)}: candidate ${str(ev.winner)} wins`, body: str(ev.ranking), ok: true }
      }
      const tests = ev.tests_passed === null || ev.tests_passed === undefined ? 'no related tests' : ev.tests_passed ? 'tests passed' : 'tests failed'
      return {
        title: `Candidate ${str(ev.index)}/${str(ev.of)}: ${tests}, ${str(ev.blocking_findings)} blocking finding(s), ${str(ev.findings)} total`,
        body: str(ev.summary),
        ok: ev.tests_passed === false ? false : null,
      }
    }
    default:
      return { title: str(ev.event_type), body: str(ev), ok: null }
  }
}

function StepLine({ ev }: { ev: StepEvent }) {
  const { title, body, ok } = describeStep(ev)
  const status = ok === null ? '' : ok ? ' step-ok' : ' step-error'
  return (
    <details className={`step-line${status}`}>
      <summary>
        <span className="step-kind">{ev.event_type}</span>
        <span className="step-title">{title}</span>
        <span className="trace-time">{new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
      </summary>
      {body && <pre className="step-body">{body}</pre>}
    </details>
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

  const nodeEvents = events.filter(isNodeEvent)
  const steps = events.filter(isStepEvent)
  const latest = nodeEvents[nodeEvents.length - 1]
  // The engine's `final` event carries the outcome (and the git results); until it arrives the
  // graph's own terminal phase still means "finishing" (commit/push/PR may be in flight).
  const phase = latest?.phase ?? 'planning'
  const isDone = latest ? Boolean(latest.final) || (TERMINAL_PHASES.has(phase) && !connected) : false
  const finalEvent = nodeEvents.find((ev) => ev.final)

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
                  i < (latest.current_plan_step ?? 0) ? 'step-done' : i === (latest.current_plan_step ?? 0) ? 'step-active' : ''
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
          <span className="stat-value">{latest?.files_changed?.length ?? 0}</span>
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
            {latest && (latest.test_total_count ?? 0) > 0
              ? `${latest.test_pass_count}/${latest.test_total_count} passed`
              : '—'}
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
          {finalEvent?.pr_url && (
            <p>
              Pull request:{' '}
              <a href={finalEvent.pr_url} target="_blank" rel="noreferrer">
                #{finalEvent.pr_number} {finalEvent.pr_url}
              </a>
            </p>
          )}
          {finalEvent?.commit_sha && (
            <p>
              Commit <code>{finalEvent.commit_sha.slice(0, 12)}</code> on <code>{finalEvent.branch_name}</code>
            </p>
          )}
        </div>
      )}

      <div className="steps-card">
        <h3>Live steps ({steps.length})</h3>
        <p className="steps-hint">
          Every tool call, tool result, test run, quality finding and model message, in full, as it happens.
        </p>
        <div className="steps-scroll">
          {steps.length === 0 && <p className="steps-empty">Waiting for the first step…</p>}
          {steps.map((ev, i) => (
            <StepLine key={i} ev={ev} />
          ))}
        </div>
      </div>

      {connectionError && <p className="error-text">Stream error: {connectionError}</p>}

      <details className="trace-log">
        <summary>Raw execution trace ({events.length} events)</summary>
        <div className="trace-scroll" ref={scrollRef}>
          {events.map((ev, i) => (
            <div key={i} className="trace-line">
              <span className="trace-node">{ev.node}</span>
              <span className="trace-phase">{ev.phase}</span>
              <span className="trace-kind">{ev.event_type ?? 'node'}</span>
              <span className="trace-time">{new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}
