// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The extension's three settings: `keystone.baseUrl` and `keystone.defaultModel` are ordinary
// configuration; the API key is NOT — it lives in VS Code's SecretStorage (the OS keychain, never
// settings.json or a synced file) under the `keystone.setApiKey` command.

import * as vscode from 'vscode'
import { KeystoneClient } from './api'

export const API_KEY_SECRET = 'keystone.apiKey'

export function baseUrl(): string {
  const configured = vscode.workspace.getConfiguration('keystone').get<string>('baseUrl') ?? 'http://localhost:8080'
  return configured.trim().replace(/\/+$/, '') || 'http://localhost:8080'
}

export function defaultModel(): string {
  return vscode.workspace.getConfiguration('keystone').get<string>('defaultModel') ?? 'auto'
}

export function getApiKey(context: vscode.ExtensionContext): Thenable<string | undefined> {
  return context.secrets.get(API_KEY_SECRET)
}

export async function storeApiKey(context: vscode.ExtensionContext, key: string): Promise<void> {
  const trimmed = key.trim()
  if (trimmed) await context.secrets.store(API_KEY_SECRET, trimmed)
  else await context.secrets.delete(API_KEY_SECRET)
}

/** Prompt for the key (masked input) and store it. Returns the stored key, or undefined if cancelled. */
export async function promptForApiKey(context: vscode.ExtensionContext): Promise<string | undefined> {
  const key = await vscode.window.showInputBox({
    title: 'Keystone API key',
    prompt: `The ks-... key for ${baseUrl()} (needs the agent scope; inference too for the model picker). Stored in VS Code's secret storage.`,
    password: true,
    ignoreFocusOut: true,
    validateInput: (value) => (value.trim() ? undefined : 'An API key is required'),
  })
  if (key === undefined) return undefined
  await storeApiKey(context, key)
  return key.trim()
}

/** A client for the configured gateway, prompting for the API key once if none is stored. */
export async function buildClient(context: vscode.ExtensionContext): Promise<KeystoneClient | undefined> {
  let apiKey = await getApiKey(context)
  if (!apiKey) {
    const choice = await vscode.window.showWarningMessage('Keystone: no API key is set for this gateway.', 'Set API key')
    if (choice !== 'Set API key') return undefined
    apiKey = await promptForApiKey(context)
    if (!apiKey) return undefined
  }
  return new KeystoneClient({ baseUrl: baseUrl(), apiKey })
}

/** A client only if a key is already stored — for background work (the status bar) that must never prompt. */
export async function quietClient(context: vscode.ExtensionContext): Promise<KeystoneClient | undefined> {
  const apiKey = await getApiKey(context)
  return apiKey ? new KeystoneClient({ baseUrl: baseUrl(), apiKey }) : undefined
}
