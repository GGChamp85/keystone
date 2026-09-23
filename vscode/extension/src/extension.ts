// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Keystone Agents for VS Code: submit a background coding task to a self-hosted Keystone
// deployment, watch its live trace (plan → tool calls → quality gates → review → tests → diff →
// PR) in a panel, list the team's recent tasks, search the team memory, open the Playground.
// Everything goes through the same REST API the web UI uses (web/src/api.ts) with one ks-... key.

import * as vscode from 'vscode'
import { type AgentTaskSummary, KeystoneClient, type TaskSubmitRequest, type TaskSubmitResponse } from './api'
import { gitRemoteUrl, normaliseRemote } from './git'
import { fallbackPickItems, type ModelPickItem, orderModelsForPicker } from './models'
import { baseUrl, buildClient, defaultModel, promptForApiKey, quietClient, storeApiKey } from './settings'
import { RunningTasksStatusBar } from './status'
import { TracePanel } from './trace/panel'

/** What `vscode.extensions.getExtension('keystone.keystone-agents').exports` offers — used by the integration test. */
export interface KeystoneExtensionApi {
  setApiKey(key: string): Promise<void>
  submitTask(req: TaskSubmitRequest): Promise<TaskSubmitResponse>
  openTrace(taskId: string): Promise<TracePanel>
  client(): Promise<KeystoneClient | undefined>
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function activate(context: vscode.ExtensionContext): KeystoneExtensionApi {
  const statusBar = new RunningTasksStatusBar(() => quietClient(context))
  context.subscriptions.push(statusBar)
  statusBar.start()

  const register = (command: string, handler: (...args: unknown[]) => Promise<void>): void => {
    context.subscriptions.push(
      vscode.commands.registerCommand(command, async (...args: unknown[]) => {
        try {
          await handler(...args)
        } catch (err) {
          void vscode.window.showErrorMessage(`Keystone: ${err instanceof Error ? err.message : String(err)}`)
        }
      }),
    )
  }

  register('keystone.setApiKey', async () => {
    const key = await promptForApiKey(context)
    if (key) {
      void vscode.window.showInformationMessage(`Keystone: API key stored for ${baseUrl()}.`)
      await statusBar.refresh()
    }
  })

  register('keystone.submitTask', async () => {
    const client = await buildClient(context)
    if (!client) return
    const taskId = await submitTaskInteractively(client)
    if (!taskId) return
    await statusBar.refresh()
    TracePanel.show(context.extensionUri, client, taskId)
    void vscode.window.showInformationMessage(`Keystone: task ${taskId.slice(0, 8)} submitted — the trace is streaming.`)
  })

  register('keystone.openTrace', async (arg) => {
    const client = await buildClient(context)
    if (!client) return
    const taskId =
      typeof arg === 'string' && UUID.test(arg)
        ? arg
        : await vscode.window.showInputBox({
            title: 'Open Keystone task trace',
            prompt: 'The task id (UUID) — from a submission, the task list, or the web UI',
            validateInput: (v) => (UUID.test(v.trim()) ? undefined : 'A task id is a UUID'),
          })
    if (!taskId) return
    TracePanel.show(context.extensionUri, client, taskId.trim())
  })

  register('keystone.listTasks', async () => {
    const client = await buildClient(context)
    if (!client) return
    const tasks = await client.listTasks({ limit: 50 })
    if (tasks.length === 0) {
      void vscode.window.showInformationMessage(`Keystone: no tasks yet at ${client.baseUrl}.`)
      return
    }
    const picked = await vscode.window.showQuickPick(tasks.map(taskPickItem), {
      title: 'Keystone: recent tasks',
      placeHolder: 'Pick a task to open its trace',
      matchOnDescription: true,
      matchOnDetail: true,
    })
    if (!picked) return
    TracePanel.show(context.extensionUri, client, picked.task.id)
  })

  register('keystone.memorySearch', async () => {
    const client = await buildClient(context)
    if (!client) return
    const query = await vscode.window.showInputBox({
      title: 'Search Keystone team memory',
      prompt: 'What the agent would recall for this query — conventions, preferences, facts, things to avoid',
      validateInput: (v) => (v.trim() ? undefined : 'Enter a query'),
    })
    if (!query) return
    const repository = normaliseRemote((await workspaceRemote()) ?? '')
    const records = await client.recallMemories(query.trim(), repository || undefined)
    if (records.length === 0) {
      void vscode.window.showInformationMessage(`Keystone: nothing recalled for "${query.trim()}".`)
      return
    }
    const picked = await vscode.window.showQuickPick(
      records.map((r) => ({
        label: `${r.pinned ? '$(pinned) ' : ''}${r.kind}`,
        description: `${r.scope}${r.repository ? ` · ${r.repository}` : ''} · ${r.status}`,
        detail: r.content,
        record: r,
      })),
      { title: `Keystone memory: ${records.length} recalled`, placeHolder: 'Select to copy the memory text', matchOnDetail: true },
    )
    if (!picked) return
    await vscode.env.clipboard.writeText(picked.record.content)
    void vscode.window.showInformationMessage('Keystone: memory copied to the clipboard.')
  })

  register('keystone.openPlayground', async () => {
    await vscode.env.openExternal(vscode.Uri.parse(`${baseUrl()}/app/?view=playground`))
  })

  return {
    setApiKey: (key) => storeApiKey(context, key),
    submitTask: async (req) => {
      const client = await quietClient(context)
      if (!client) throw new Error('no API key stored — call setApiKey first')
      const resp = await client.submitTask(req)
      await statusBar.refresh()
      return resp
    },
    openTrace: async (taskId) => {
      const client = await quietClient(context)
      if (!client) throw new Error('no API key stored — call setApiKey first')
      return TracePanel.show(context.extensionUri, client, taskId)
    },
    client: () => quietClient(context),
  }
}

export function deactivate(): void {
  // every panel, the status bar and the commands are context.subscriptions — disposed by the host
}

async function workspaceRemote(): Promise<string | undefined> {
  const folder = vscode.workspace.workspaceFolders?.[0]
  return folder ? gitRemoteUrl(folder.uri.fsPath) : undefined
}

/** description → repository → model, each step cancellable; returns the new task id. */
async function submitTaskInteractively(client: KeystoneClient): Promise<string | undefined> {
  const task = await vscode.window.showInputBox({
    title: 'Keystone: submit a background task (1/3)',
    prompt: 'What should the agent do? It clones the repo into a sandbox, edits with real tools, runs the quality gates and tests, and opens a PR.',
    placeHolder: 'Add cursor-based pagination to the /users endpoint',
    ignoreFocusOut: true,
    validateInput: (v) => (v.trim().length >= 10 ? undefined : 'At least 10 characters (AgentTaskRequest.task)'),
  })
  if (!task) return undefined

  const remote = await workspaceRemote()
  const repositoryUrl = await vscode.window.showInputBox({
    title: 'Keystone: repository (2/3)',
    prompt: 'Git clone URL the agent works on — the gateway must allow its host (GIT_ALLOWED_HOSTS). Leave empty for a repository-less task.',
    value: remote ? normaliseRemote(remote) : '',
    ignoreFocusOut: true,
  })
  if (repositoryUrl === undefined) return undefined

  const model = await pickModel(client)
  if (!model) return undefined

  const resp = await client.submitTask({
    task: task.trim(),
    repository_url: repositoryUrl.trim() || undefined,
    model,
  })
  return resp.task_id
}

type ModelQuickPickItem = vscode.QuickPickItem & { item?: ModelPickItem }

async function pickModel(client: KeystoneClient): Promise<string | undefined> {
  const preferred = defaultModel()
  let items: ModelPickItem[]
  let note = ''
  try {
    items = orderModelsForPicker(await client.getModelLibrary(), preferred)
  } catch (err) {
    items = fallbackPickItems(preferred)
    note = ` — model library unavailable (${err instanceof Error ? err.message : String(err)})`
  }
  const quickItems: ModelQuickPickItem[] = []
  let separated = false
  for (const item of items) {
    if (!item.selectable && !separated) {
      quickItems.push({ label: 'not ready', kind: vscode.QuickPickItemKind.Separator })
      separated = true
    }
    quickItems.push({ label: item.label, description: item.description, detail: item.detail, item })
  }
  while (true) {
    const picked = await vscode.window.showQuickPick(quickItems, {
      title: `Keystone: model (3/3)${note}`,
      placeHolder: 'A gateway role, or auto to let the router decide',
      ignoreFocusOut: true,
    })
    if (!picked?.item) return undefined
    if (picked.item.selectable) return picked.item.id
    void vscode.window.showWarningMessage(
      `Keystone: ${picked.item.id} is ${picked.item.state} — pick a READY role or auto. See the web UI's Models view for why.`,
    )
  }
}

const STATUS_ICON: Record<string, string> = {
  pending: '$(clock)',
  running: '$(sync~spin)',
  completed: '$(pass-filled)',
  failed: '$(error)',
  cancelled: '$(circle-slash)',
  timed_out: '$(watch)',
}

function taskPickItem(t: AgentTaskSummary): vscode.QuickPickItem & { task: AgentTaskSummary } {
  const description = t.task_description.length > 80 ? `${t.task_description.slice(0, 77)}…` : t.task_description
  const parts = [t.model_role, t.repository_url ?? 'no repository']
  if (t.pr_url) parts.push(`PR #${t.pr_number ?? '?'}`)
  else if (t.branch_name) parts.push(t.branch_name)
  if (t.error_message) parts.push(t.error_message)
  return {
    label: `${STATUS_ICON[t.status] ?? '$(question)'} ${t.status}`,
    description,
    detail: `${new Date(t.created_at).toLocaleString()} · ${parts.join(' · ')}`,
    task: t,
  }
}
