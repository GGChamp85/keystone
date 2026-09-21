---
description: Inspect and manage Keystone's memory for this repo/team
---

Run this command with the `keystone` CLI (see docs/deployment/OPENCODE_SETUP.md — it needs
KEYSTONE_INFERENCE_URL and KEYSTONE_API_KEY set the same way this OpenCode session's provider is):

!`keystone memory $ARGUMENTS`

If the command above failed because `keystone` isn't installed or the env vars aren't set, tell the user
exactly what's missing and how to fix it (`pip install -e .` from the Keystone repo root, or see the README's
Quick Start) — don't try to work around it by calling the API directly.

Otherwise, summarize the output above in plain language: what memories exist, their scope (repo vs team-wide)
and status (proposed memories need `keystone memory approve <id>` before they're ever recalled into a task),
and if the user asked to add/forget/pin something, confirm what changed.
