// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// How a task event is described for the trace panel — a port of web/src/components/TaskStreamView.tsx's
// describeStep/PhaseTimeline, kept as pure functions on the extension-host side so the webview only
// renders strings and so the descriptions can be tested against the recorded real stream.

import { isNodeEvent, type NodeEvent, type StepEvent, type TaskEvent, TERMINAL_PHASES } from '../api'

export const PHASE_ORDER = ['planning', 'coding', 'quality', 'review', 'testing', 'fixing', 'complete'] as const

export const PHASE_LABELS: Record<string, string> = {
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

export type PhaseState = 'done' | 'active' | 'pending'

export interface PhaseStep {
  phase: string
  label: string
  state: PhaseState
}

/** The timeline's per-phase state for the current phase, exactly as TaskStreamView draws it. */
export function phaseTimeline(current: string): { steps: PhaseStep[]; failed: boolean } {
  const currentIndex = (PHASE_ORDER as readonly string[]).indexOf(current)
  const failed = current === 'failed' || current === 'cancelled'
  const steps = PHASE_ORDER.map((phase, i) => {
    let state: PhaseState = 'pending'
    if (failed) {
      state = i < currentIndex ? 'done' : 'pending'
    } else if (currentIndex >= 0) {
      if (i < currentIndex) state = 'done'
      else if (i === currentIndex) state = 'active'
    }
    return { phase, label: PHASE_LABELS[phase], state }
  })
  return { steps, failed }
}

export function str(v: unknown): string {
  if (v === null || v === undefined) return ''
  return typeof v === 'string' ? v : JSON.stringify(v, null, 2)
}

export interface DescribedStep {
  kind: string
  title: string
  body: string
  ok: boolean | null
  timestamp: number
}

export function describeStep(ev: StepEvent): DescribedStep {
  const base = { kind: ev.event_type, timestamp: ev.timestamp }
  switch (ev.event_type) {
    case 'tool_call':
      return { ...base, title: `${str(ev.name)}(${str(ev.arguments)})`, body: str(ev.error), ok: ev.error ? false : null }
    case 'tool_result':
      return {
        ...base,
        title: `${str(ev.name)} → ${ev.ok ? 'ok' : 'error'}`,
        body: ev.ok ? str(ev.output) : str(ev.error),
        ok: Boolean(ev.ok),
      }
    case 'model_text':
      return { ...base, title: ev.final ? 'Model: final message' : 'Model', body: str(ev.content), ok: null }
    case 'test_output':
      return {
        ...base,
        title: `${str(ev.name)} → exit ${str(ev.exit_code)} (${ev.passed ? 'passed' : 'failed'})`,
        body: [str(ev.stdout), str(ev.stderr)].filter(Boolean).join('\n--- stderr ---\n'),
        ok: Boolean(ev.passed),
      }
    case 'quality_findings': {
      const findings = (ev.findings as unknown[]) ?? []
      const blocking = (ev.blocking as unknown[]) ?? []
      return {
        ...base,
        title: `${findings.length} finding(s), ${blocking.length} blocking`,
        body: str(findings),
        ok: blocking.length === 0,
      }
    }
    case 'deps_install':
      return { ...base, title: `${str(ev.command)} → ${ev.ok ? 'ok' : 'failed'}`, body: str(ev.output), ok: Boolean(ev.ok) }
    case 'repo_map':
      return { ...base, title: 'Repository map given to the planner and coder', body: str(ev.map), ok: null }
    case 'route_decision':
      return { ...base, title: `model=auto → ${str(ev.resolved_role)}`, body: str(ev), ok: null }
    case 'diff':
      return {
        ...base,
        title: `Committed ${str(ev.commit_sha).slice(0, 12)} on ${str(ev.branch_name)}`,
        body: str(ev.diff),
        ok: true,
      }
    case 'pr':
      return { ...base, title: `Pull request #${str(ev.pr_number)}: ${str(ev.pr_url)}`, body: '', ok: true }
    default:
      return { ...base, title: str(ev.event_type), body: str(ev), ok: null }
  }
}

export interface TraceSummary {
  phase: string
  isDone: boolean
  latest: NodeEvent | null
  final: NodeEvent | null
  stepCount: number
  eventCount: number
}

/**
 * The header numbers TaskStreamView derives from the node events: the current phase, whether the
 * engine's `final` event has arrived (until then the graph's own terminal phase still means
 * "finishing" — commit/push/PR may be in flight), and the latest per-node summary.
 */
export function summarize(events: TaskEvent[], connected: boolean): TraceSummary {
  const nodeEvents = events.filter(isNodeEvent)
  const latest = nodeEvents[nodeEvents.length - 1] ?? null
  const phase = latest?.phase ?? 'planning'
  const isDone = latest ? Boolean(latest.final) || (TERMINAL_PHASES.has(phase) && !connected) : false
  return {
    phase,
    isDone,
    latest,
    final: nodeEvents.find((ev) => ev.final) ?? null,
    stepCount: events.length - nodeEvents.length,
    eventCount: events.length,
  }
}
