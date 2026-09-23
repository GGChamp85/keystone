// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The trace panel's webview script. Draws only what the extension host sends (src/trace/panel.ts's
// HostMessage): a snapshot of every event so far, then one message per new event, then the final
// connection status. All text goes through textContent — nothing from the stream is ever HTML.
// @ts-check
;(function () {
  const vscode = acquireVsCodeApi()
  const app = /** @type {HTMLElement} */ (document.getElementById('app'))
  const taskId = app.dataset.taskId || ''

  const PHASES = ['planning', 'coding', 'quality', 'review', 'testing', 'fixing', 'complete']
  const LABELS = {
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

  /** @type {{ items: any[]; summary: any; connected: boolean; error: string | null }} */
  const state = { items: [], summary: null, connected: true, error: null }

  function el(tag, className, text) {
    const node = document.createElement(tag)
    if (className) node.className = className
    if (text !== undefined && text !== null) node.textContent = String(text)
    return node
  }

  function timeOf(ts) {
    return new Date(ts * 1000).toLocaleTimeString()
  }

  function stepNode(item) {
    const step = item.step
    const details = el('details', 'step-line' + (step.ok === null ? '' : step.ok ? ' step-ok' : ' step-error'))
    const summary = el('summary')
    summary.appendChild(el('span', 'step-kind', step.kind))
    summary.appendChild(el('span', 'step-title', step.title))
    summary.appendChild(el('span', 'trace-time', timeOf(step.timestamp)))
    details.appendChild(summary)
    if (step.body) details.appendChild(el('pre', 'step-body', step.body))
    return details
  }

  const header = el('div')
  const stepsCard = el('div', 'card')
  const stepsTitle = el('h2', '', 'Live steps (0)')
  const stepsHint = el('p', 'muted', 'Every tool call, tool result, test run, quality finding and model message, in full, as it happens.')
  const stepsEmpty = el('p', 'muted', 'Waiting for the first step…')
  const stepsList = el('div')
  stepsCard.append(stepsTitle, stepsHint, stepsEmpty, stepsList)
  const statusLine = el('p', 'status-line muted')
  let stepCount = 0

  function renderHeader() {
    header.replaceChildren()
    const summary = state.summary
    const latest = summary ? summary.latest : null
    const phase = summary ? summary.phase : 'planning'
    const isDone = summary ? summary.isDone : false
    const finalEvent = summary ? summary.final : null

    const h1 = el('h1', '', 'Task ')
    h1.appendChild(el('code', '', taskId))
    header.appendChild(h1)

    const timeline = el('div', 'phase-timeline')
    const currentIndex = PHASES.indexOf(phase)
    const failed = phase === 'failed' || phase === 'cancelled'
    PHASES.forEach((p, i) => {
      let cls = 'pending'
      if (failed) cls = i < currentIndex ? 'done' : 'pending'
      else if (currentIndex >= 0) cls = i < currentIndex ? 'done' : i === currentIndex ? 'active' : 'pending'
      const step = el('div', 'phase-step phase-' + cls)
      step.appendChild(el('span', 'phase-dot'))
      step.appendChild(el('span', 'phase-label', LABELS[p]))
      timeline.appendChild(step)
    })
    if (failed) timeline.appendChild(el('div', 'phase-step phase-active phase-failed', LABELS[phase]))
    header.appendChild(timeline)

    if (latest && latest.plan_steps && latest.plan_steps.length > 0) {
      const plan = el('div', 'card')
      plan.appendChild(el('h2', '', 'Plan'))
      if (latest.plan) plan.appendChild(el('p', '', latest.plan))
      const ol = el('ol', 'plan-steps')
      const current = latest.current_plan_step || 0
      latest.plan_steps.forEach((text, i) => {
        ol.appendChild(el('li', i < current ? 'step-done' : i === current ? 'step-active' : '', text))
      })
      plan.appendChild(ol)
      header.appendChild(plan)
    }

    const stats = el('div', 'stats-grid')
    const stat = (label, value) => {
      const box = el('div', 'stat')
      box.appendChild(el('span', 'stat-label', label))
      box.appendChild(el('span', 'stat-value', value))
      stats.appendChild(box)
    }
    stat('Iteration', `${latest ? latest.iteration || 0 : 0}${latest && latest.max_iterations ? ` / ${latest.max_iterations}` : ''}`)
    stat('Files changed', latest && latest.files_changed ? latest.files_changed.length : 0)
    stat(
      'Review',
      !latest || latest.review_passed === null || latest.review_passed === undefined
        ? '—'
        : latest.review_passed
          ? 'Passed'
          : `${latest.review_comment_count} comment(s)`,
    )
    stat('Tests', latest && (latest.test_total_count || 0) > 0 ? `${latest.test_pass_count}/${latest.test_total_count} passed` : '—')
    stat('Tokens used', latest && latest.total_tokens ? latest.total_tokens.toLocaleString() : 0)
    stat('Status', state.connected ? 'Live' : isDone ? 'Finished' : 'Disconnected')
    header.appendChild(stats)

    if (latest && latest.files_changed && latest.files_changed.length > 0) {
      const files = el('div', 'card')
      files.appendChild(el('h2', '', 'Files touched'))
      const ul = el('ul')
      latest.files_changed.forEach((f) => {
        const li = el('li')
        li.appendChild(el('code', '', f))
        ul.appendChild(li)
      })
      files.appendChild(ul)
      header.appendChild(files)
    }

    if (isDone && latest) {
      const result = el('div', 'card ' + (phase === 'complete' ? 'result-ok' : 'result-error'))
      result.appendChild(el('h2', '', phase === 'complete' ? 'Result' : 'Failure'))
      result.appendChild(el('p', '', latest.result_summary || latest.error_message || 'No summary available.'))
      if (finalEvent && finalEvent.pr_url) {
        const p = el('p', '', 'Pull request: ')
        const a = el('a', '', `#${finalEvent.pr_number} ${finalEvent.pr_url}`)
        a.href = finalEvent.pr_url
        p.appendChild(a)
        result.appendChild(p)
      }
      if (finalEvent && finalEvent.commit_sha) {
        const p = el('p', '', 'Commit ')
        p.appendChild(el('code', '', String(finalEvent.commit_sha).slice(0, 12)))
        p.appendChild(document.createTextNode(' on '))
        p.appendChild(el('code', '', finalEvent.branch_name || ''))
        result.appendChild(p)
      }
      header.appendChild(result)
    }
  }

  function renderStatus() {
    statusLine.className = 'status-line ' + (state.error ? 'error-text' : 'muted')
    statusLine.textContent = state.error
      ? `Stream error: ${state.error}`
      : state.connected
        ? 'Connected — events stream in as the agent works.'
        : 'The stream closed.'
  }

  function appendItem(item) {
    if (item.step) {
      stepCount += 1
      stepsEmpty.hidden = true
      stepsList.appendChild(stepNode(item))
      stepsTitle.textContent = `Live steps (${stepCount})`
    }
  }

  function mount() {
    app.replaceChildren(header, stepsCard, statusLine)
  }

  window.addEventListener('message', (event) => {
    const msg = event.data
    if (!msg || typeof msg.type !== 'string') return
    if (msg.type === 'snapshot') {
      state.items = msg.items
      state.summary = msg.summary
      state.connected = msg.connected
      state.error = msg.error
      stepsList.replaceChildren()
      stepCount = 0
      stepsEmpty.hidden = false
      stepsTitle.textContent = 'Live steps (0)'
      mount()
      msg.items.forEach(appendItem)
      renderHeader()
      renderStatus()
    } else if (msg.type === 'event') {
      state.items.push(msg.item)
      state.summary = msg.summary
      appendItem(msg.item)
      renderHeader()
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    } else if (msg.type === 'status') {
      state.connected = msg.connected
      state.error = msg.error
      state.summary = msg.summary
      renderHeader()
      renderStatus()
    }
  })

  vscode.postMessage({ type: 'ready' })
})()
