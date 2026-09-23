// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Status bar item: how many of this tenant's tasks are pending or running right now, from
// GET /v1/keystone/tasks (the team task list — every task in the tenant, not just this key's).
// Polls while an API key is stored; never prompts. Click → the recent-tasks picker.

import * as vscode from 'vscode'
import type { KeystoneClient } from './api'
import { countActive } from './tasks'

export const POLL_INTERVAL_MS = 15_000

export class RunningTasksStatusBar implements vscode.Disposable {
  private readonly item: vscode.StatusBarItem
  private timer: NodeJS.Timeout | undefined
  private refreshing = false

  constructor(private readonly getClient: () => Promise<KeystoneClient | undefined>) {
    this.item = vscode.window.createStatusBarItem('keystone.runningTasks', vscode.StatusBarAlignment.Left, 50)
    this.item.name = 'Keystone running tasks'
    this.item.command = 'keystone.listTasks'
    this.item.text = '$(rocket) Keystone'
    this.item.tooltip = 'Keystone: set an API key to see running tasks'
    this.item.show()
  }

  start(): void {
    void this.refresh()
    this.timer = setInterval(() => void this.refresh(), POLL_INTERVAL_MS)
  }

  async refresh(): Promise<void> {
    if (this.refreshing) return
    this.refreshing = true
    try {
      const client = await this.getClient()
      if (!client) {
        this.item.text = '$(rocket) Keystone'
        this.item.tooltip = 'Keystone: set an API key to see running tasks'
        return
      }
      const tasks = await client.listTasks({ limit: 200 })
      const active = countActive(tasks)
      this.item.text = `$(rocket) Keystone: ${active} running`
      this.item.tooltip = `${active} task(s) pending or running of the last ${tasks.length} at ${client.baseUrl} — click to list`
      this.item.backgroundColor = undefined
    } catch (err) {
      this.item.text = '$(warning) Keystone'
      this.item.tooltip = `Keystone: ${err instanceof Error ? err.message : String(err)}`
      this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground')
    } finally {
      this.refreshing = false
    }
  }

  dispose(): void {
    if (this.timer) clearInterval(this.timer)
    this.item.dispose()
  }
}
