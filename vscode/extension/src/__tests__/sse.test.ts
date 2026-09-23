// The SSE decoder against the real recorded stream (fixtures/task-stream.sse — the exact bytes
// GET /v1/keystone/tasks/{id}/stream sent through the real app; scripts/record_task_stream_fixture.py),
// fed at every chunk size from one byte up, plus the route's real keep-alive comment.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { isNodeEvent, isStepEvent, type NodeEvent, type StepEventType, type TaskEvent, TERMINAL_PHASES } from '../api'
import { parsePayloads, SseFrameDecoder } from '../sse'

const FIXTURE = readFileSync(join(__dirname, 'fixtures', 'task-stream.sse'))
const KEEP_ALIVE = ': keep-alive\n\n' // src/api/routes/agents.py's stream_task, verbatim

const EXPECTED_ORDER: (StepEventType | 'node')[] = [
  'route_decision',
  'deps_install',
  'repo_map',
  'node',
  'tool_call',
  'tool_result',
  'tool_call',
  'tool_result',
  'model_text',
  'node',
  'quality_findings',
  'node',
  'node',
  'test_output',
  'test_output',
  'node',
  'diff',
  'pr',
  'node',
]

function decodeAll(chunkSize: number): TaskEvent[] {
  const decoder = new SseFrameDecoder()
  const events: TaskEvent[] = []
  for (let offset = 0; offset < FIXTURE.length; offset += chunkSize) {
    events.push(...parsePayloads<TaskEvent>(decoder.push(FIXTURE.subarray(offset, offset + chunkSize))))
  }
  expect(decoder.pending).toBe('') // the recorded stream ends on a complete frame
  return events
}

describe('SseFrameDecoder on the recorded real stream', () => {
  it('yields every frame in order when fed the whole body at once', () => {
    const events = decodeAll(FIXTURE.length)
    expect(events.map((e) => e.event_type ?? 'node')).toEqual(EXPECTED_ORDER)
  })

  it.each([1, 7, 64, 1024])('yields the same events when chunked at %d bytes', (size) => {
    expect(decodeAll(size)).toEqual(decodeAll(FIXTURE.length))
  })

  it('a multi-byte character split across chunks survives (TextDecoder stream mode)', () => {
    const events = decodeAll(3)
    const failed = events[5]
    expect(isStepEvent(failed) && failed.event_type === 'tool_result' && failed.ok === false).toBe(true)
    // the real apply_patch error text carries an em dash (U+2014, three UTF-8 bytes)
    expect(String((failed as { error?: unknown }).error)).toContain('Re-read the file — it may have changed')
  })

  it('carries the full step payloads the trace renders', () => {
    const events = decodeAll(FIXTURE.length)
    const steps = events.filter(isStepEvent)
    const byType = (t: StepEventType) => steps.filter((s) => s.event_type === t)
    expect(byType('tool_call')[0]).toMatchObject({ node: 'coding', phase: 'coding', turn: 0, name: 'apply_patch' })
    expect(byType('tool_call')[0].arguments).toEqual({ path: 'app.py', search: 'NOPE NOT THERE', replace: 'x' })
    expect(byType('tool_result')[1]).toMatchObject({ ok: true, output: 'Updated app.py (41 bytes)', error: '' })
    expect(byType('model_text')[0]).toMatchObject({ final: true, content: "Fixed after the first search didn't match." })
    expect(byType('test_output').map((s) => s.name)).toEqual(['related', 'all'])
    expect(String(byType('test_output')[0].stdout)).toContain('2 passed')
    expect((byType('quality_findings')[0].findings as unknown[]).length).toBe(1)
    expect(byType('quality_findings')[0].blocking).toEqual([])
    expect(String(byType('diff')[0].diff)).toContain('+    return f"HI {name}"')
    expect(byType('pr')[0]).toMatchObject({ pr_number: 42, phase: 'complete', node: 'finalize' })
    expect(byType('route_decision')[0]).toMatchObject({ requested: 'auto', resolved_role: 'coding', phase: 'pending' })
    for (const s of steps) expect(typeof s.timestamp).toBe('number')
  })

  it('ends on the engine final node event, after the diff and the PR', () => {
    const events = decodeAll(FIXTURE.length)
    const last = events[events.length - 1] as NodeEvent
    expect(isNodeEvent(last)).toBe(true)
    expect(last.final).toBe(true)
    expect(last.node).toBe('finalize')
    expect(TERMINAL_PHASES.has(last.phase)).toBe(true)
    expect(last.pr_url).toBe('https://git.example.internal/acme/greeter/pulls/42')
    expect(last.commit_sha).toBe('9b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c')
    // the model's own `final: true` (model_text) must NOT be mistaken for the end of the task
    const modelFinal = events.find((e) => isStepEvent(e) && e.event_type === 'model_text')
    expect(modelFinal && (modelFinal as { final?: unknown }).final).toBe(true)
    expect(events.indexOf(modelFinal!)).toBeLessThan(events.length - 1)
  })

  it('a node event carries the per-node summary fields', () => {
    const events = decodeAll(FIXTURE.length)
    const first = events.find(isNodeEvent)!
    expect(first).toMatchObject({ event_type: 'node', node: 'planning', phase: 'coding', iteration: 1, max_iterations: 15 })
    expect(first.plan_steps).toEqual(['Read app.py', 'Patch greet()', 'Run the tests'])
    const tested = events.filter(isNodeEvent).find((e) => e.node === 'testing')!
    expect(tested).toMatchObject({ tests_passed: true, test_pass_count: 2, test_total_count: 2, review_passed: true })
    expect(tested.files_changed).toEqual(['app.py'])
    expect(tested.total_tokens).toBe(2210 + 402)
  })
})

describe('SseFrameDecoder edge cases', () => {
  it('skips the route keep-alive comment frames and keeps decoding after them', () => {
    const decoder = new SseFrameDecoder()
    const text = FIXTURE.toString('utf8')
    const frames = text.split('\n\n').filter(Boolean)
    const interleaved = frames.map((f) => `${f}\n\n${KEEP_ALIVE}`).join('')
    const payloads = decoder.push(KEEP_ALIVE + interleaved)
    expect(payloads.length).toBe(frames.length)
    expect(parsePayloads<TaskEvent>(payloads).map((e) => e.event_type ?? 'node')).toEqual(EXPECTED_ORDER)
  })

  it('holds an incomplete frame until its blank line arrives', () => {
    const decoder = new SseFrameDecoder()
    expect(decoder.push('id: 1-0\ndata: {"event_type": "node", "node": "planning", "phase": "coding"')).toEqual([])
    expect(decoder.pending).toContain('data: ')
    const [payload] = decoder.push(', "timestamp": 1.0}\n\n')
    expect(JSON.parse(payload)).toMatchObject({ node: 'planning', phase: 'coding' })
    expect(decoder.pending).toBe('')
  })

  it('a malformed payload is skipped without losing the frames around it', () => {
    const decoder = new SseFrameDecoder()
    const payloads = decoder.push('data: {"a": 1}\n\ndata: {not json\n\ndata: {"b": 2}\n\n')
    expect(payloads).toHaveLength(3)
    expect(parsePayloads<Record<string, number>>(payloads)).toEqual([{ a: 1 }, { b: 2 }])
  })
})
