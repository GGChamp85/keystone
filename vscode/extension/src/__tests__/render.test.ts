// The step descriptions and header summary the trace panel shows, computed over the recorded real
// stream — the same titles web/src/components/TaskStreamView.tsx renders for the same events.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { isStepEvent, type StepEvent, type TaskEvent } from '../api'
import { parsePayloads, SseFrameDecoder } from '../sse'
import { describeStep, phaseTimeline, summarize } from '../trace/render'

const EVENTS = parsePayloads<TaskEvent>(new SseFrameDecoder().push(readFileSync(join(__dirname, 'fixtures', 'task-stream.sse'))))
const STEPS = EVENTS.filter(isStepEvent)
const step = (type: string, n = 0): StepEvent => STEPS.filter((s) => s.event_type === type)[n]

describe('describeStep', () => {
  it('tool calls and results', () => {
    expect(describeStep(step('tool_call', 0))).toMatchObject({
      kind: 'tool_call',
      title: 'apply_patch({\n  "path": "app.py",\n  "search": "NOPE NOT THERE",\n  "replace": "x"\n})',
      body: '',
      ok: null,
    })
    const failed = describeStep(step('tool_result', 0))
    expect(failed).toMatchObject({ title: 'apply_patch → error', ok: false })
    expect(failed.body).toContain('was not found')
    expect(describeStep(step('tool_result', 1))).toMatchObject({ title: 'apply_patch → ok', body: 'Updated app.py (41 bytes)', ok: true })
  })

  it('the model message, the repo map, the install and the routing decision', () => {
    expect(describeStep(step('model_text'))).toMatchObject({ title: 'Model: final message', body: "Fixed after the first search didn't match.", ok: null })
    expect(describeStep(step('repo_map')).body).toContain('function greet(name)')
    expect(describeStep(step('deps_install'))).toMatchObject({ title: 'pip install -r requirements.txt → ok', ok: true })
    expect(describeStep(step('route_decision')).title).toBe('model=auto → coding')
  })

  it('quality findings, test output, diff and PR', () => {
    const quality = describeStep(step('quality_findings'))
    expect(quality).toMatchObject({ title: '1 finding(s), 0 blocking', ok: true })
    expect(quality.body).toContain('Missing docstring in public function')
    expect(describeStep(step('test_output', 0))).toMatchObject({ title: 'related → exit 0 (passed)', ok: true })
    expect(describeStep(step('test_output', 1)).body).toContain('2 passed in 0.03s')
    expect(describeStep(step('test_output', 1)).body).not.toContain('--- stderr ---') // empty stderr is not shown
    expect(describeStep(step('diff'))).toMatchObject({ title: 'Committed 9b1c2d3e4f5a on keystone/ada/3f2a9c1e', ok: true })
    expect(describeStep(step('diff')).body).toContain('diff --git a/app.py b/app.py')
    expect(describeStep(step('pr'))).toMatchObject({ title: 'Pull request #42: https://git.example.internal/acme/greeter/pulls/42', body: '', ok: true })
  })

  it('a failed test run and a blocking finding are marked as errors', () => {
    const failing: StepEvent = { ...step('test_output', 0), exit_code: 1, passed: false, stderr: 'E   assert 1 == 2' }
    const described = describeStep(failing)
    expect(described).toMatchObject({ title: 'related → exit 1 (failed)', ok: false })
    expect(described.body).toContain('--- stderr ---\nE   assert 1 == 2')
    const blocking: StepEvent = { ...step('quality_findings'), blocking: [{ tool: 'bandit' }] }
    expect(describeStep(blocking)).toMatchObject({ title: '1 finding(s), 1 blocking', ok: false })
  })
})

describe('summarize + phaseTimeline', () => {
  it('follows the node events: not done until the engine final event, then done with the git results', () => {
    const beforeFinal = summarize(EVENTS.slice(0, -1), true)
    expect(beforeFinal.phase).toBe('complete') // the graph's terminal phase arrived…
    expect(beforeFinal.isDone).toBe(false) // …but commit/push/PR were still in flight
    expect(beforeFinal.final).toBeNull()

    const all = summarize(EVENTS, false)
    // 19 frames: 6 node summaries (planning, coding, quality, review, testing, finalize) + 13 steps
    expect(all).toMatchObject({ phase: 'complete', isDone: true, stepCount: 13, eventCount: 19 })
    expect(all.final?.pr_number).toBe(42)
    expect(all.latest?.node).toBe('finalize')
  })

  it('an empty stream is still planning, not done', () => {
    expect(summarize([], true)).toMatchObject({ phase: 'planning', isDone: false, latest: null, final: null, stepCount: 0 })
  })

  it('draws the timeline exactly as the web UI does', () => {
    const testing = phaseTimeline('testing')
    expect(testing.failed).toBe(false)
    expect(testing.steps.map((s) => s.state)).toEqual(['done', 'done', 'done', 'done', 'active', 'pending', 'pending'])
    const failed = phaseTimeline('failed')
    expect(failed.failed).toBe(true)
    expect(failed.steps.every((s) => s.state === 'pending')).toBe(true) // 'failed' is not on the happy-path list
    expect(phaseTimeline('complete').steps.at(-1)).toMatchObject({ label: 'Complete', state: 'active' })
  })
})
