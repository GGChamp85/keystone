// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Runs inside a real VS Code extension host (@vscode/test-electron). The activation test always
// runs. The live test submits a real task to a real Keystone gateway and asserts the trace panel
// receives real stream events — only when KEYSTONE_URL and KEYSTONE_API_KEY are set; otherwise it
// self-skips with a printed reason rather than pretending to pass.
import * as assert from 'node:assert/strict'
import * as vscode from 'vscode'
import type { KeystoneExtensionApi } from '../../extension'

const EXTENSION_ID = 'keystone.keystone-agents'
const COMMANDS = [
  'keystone.submitTask',
  'keystone.openTrace',
  'keystone.listTasks',
  'keystone.memorySearch',
  'keystone.openPlayground',
  'keystone.setApiKey',
]

async function activated(): Promise<KeystoneExtensionApi> {
  const ext = vscode.extensions.getExtension<KeystoneExtensionApi>(EXTENSION_ID)
  assert.ok(ext, `extension ${EXTENSION_ID} is not loaded`)
  return ext.activate()
}

suite('Keystone Agents extension', () => {
  test('activates and registers every command', async () => {
    await activated()
    const registered = await vscode.commands.getCommands(true)
    for (const command of COMMANDS) assert.ok(registered.includes(command), `${command} is not registered`)
  })

  test('submits a real task and the trace panel receives events (needs KEYSTONE_URL + KEYSTONE_API_KEY)', async function () {
    const url = process.env.KEYSTONE_URL
    const apiKey = process.env.KEYSTONE_API_KEY
    if (!url || !apiKey) {
      console.log('SKIPPED: set KEYSTONE_URL and KEYSTONE_API_KEY to run the live submit-and-stream test against a real gateway')
      this.skip()
      return
    }
    const api = await activated()
    await vscode.workspace.getConfiguration('keystone').update('baseUrl', url, vscode.ConfigurationTarget.Global)
    await api.setApiKey(apiKey)

    const submitted = await api.submitTask({
      task: 'Integration test from the VS Code extension: report the repository layout and make no changes.',
      model: 'auto',
    })
    assert.match(submitted.task_id, /^[0-9a-f-]{36}$/)
    assert.equal(submitted.status, 'pending')

    const panel = await api.openTrace(submitted.task_id)
    const events = await panel.waitForEvents(1, 60_000)
    assert.ok(events.length >= 1, 'no events received')
    for (const ev of events) {
      assert.equal(typeof ev.node, 'string')
      assert.equal(typeof ev.phase, 'string')
      assert.equal(typeof ev.timestamp, 'number')
    }
    // model=auto → the engine publishes its routing decision before the task even starts
    const routing = events.find((ev) => ev.event_type === 'route_decision')
    assert.ok(routing, `expected a route_decision event, got ${events.map((e) => e.event_type ?? 'node').join(', ')}`)
    assert.equal((routing as { requested?: unknown }).requested, 'auto')

    const client = await api.client()
    assert.ok(client)
    const listed = await client.listTasks({ limit: 50 })
    assert.ok(listed.some((t) => t.id === submitted.task_id), 'the submitted task is in the team task list')
    panel.dispose()
  })
})
