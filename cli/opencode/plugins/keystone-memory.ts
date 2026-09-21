// Copyright 2024-2026 Gaurav Gupta / Keystone
// Licensed under the Apache License, Version 2.0
//
// Injects recalled Keystone memory (src/memory/store.py's real
// repo-over-tenant ranking, via GET/POST /v1/keystone/memory) into every
// OpenCode session's system prompt — the same context the background
// agent's own coding/planning nodes see, so a developer working
// interactively benefits from the same learned conventions/avoid-list as
// the autonomous agent, not a separate unsynced copy of it.
//
// Verified against OpenCode's real Plugin/Hooks types (@opencode-ai/plugin,
// packages/plugin/src/index.ts in github.com/anomalyco/opencode — the
// current home of the project after it moved from sst/opencode) — the
// `experimental.chat.system.transform` hook only receives {sessionID,
// model}, not the message text, so this plugin tracks the last user
// message via the separate `chat.message` hook and uses that as the
// recall query.
//
// Auto-discovered from `.opencode/plugins/` (project- or user-level) —
// copy this file there the same way cli/opencode.config.json gets copied
// to opencode.json (see docs/deployment/OPENCODE_SETUP.md).

import type { Plugin } from "@opencode-ai/plugin"

interface MemoryRecord {
  kind: string
  scope: string
  content: string
}

export const KeystoneMemoryPlugin: Plugin = async ({ directory, $ }) => {
  const baseUrl = process.env.KEYSTONE_INFERENCE_URL
  const apiKey = process.env.KEYSTONE_API_KEY
  let lastUserText = ""

  async function repositoryUrl(): Promise<string | null> {
    try {
      const result = await $`git -C ${directory} remote get-url origin`.text()
      const trimmed = result.trim()
      return trimmed.length > 0 ? trimmed : null
    } catch {
      // Not a git repo, or no "origin" remote — recall falls back to
      // tenant-scope memory only, same as the background agent does for a
      // task with no repository_url.
      return null
    }
  }

  async function recall(query: string): Promise<string> {
    if (!baseUrl || !apiKey || query.trim().length === 0) return ""
    try {
      const repository = await repositoryUrl()
      const res = await fetch(`${baseUrl.replace(/\/$/, "")}/v1/keystone/memory/recall`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, repository }),
      })
      if (!res.ok) return ""
      const memories = (await res.json()) as MemoryRecord[]
      if (!Array.isArray(memories) || memories.length === 0) return ""
      const lines = memories.map((m) => `- [${m.kind}/${m.scope}] ${m.content}`)
      return "## Team Memory (learned conventions, preferences, and things to avoid)\n" + lines.join("\n")
    } catch {
      // Keystone unreachable — degrade silently, same fail-open-for-
      // advisory-context policy the background agent's own memory recall
      // uses (a missing recall never blocks a session).
      return ""
    }
  }

  return {
    "chat.message": async (_input, output) => {
      const textPart = output.parts.find((p): p is typeof p & { type: "text"; text: string } => p.type === "text")
      if (textPart) lastUserText = textPart.text
    },
    "experimental.chat.system.transform": async (_input, output) => {
      const memoryText = await recall(lastUserText)
      if (memoryText) output.system.push(memoryText)
    },
  }
}

export default KeystoneMemoryPlugin
