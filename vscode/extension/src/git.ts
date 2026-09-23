// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The workspace's git remote, used as the default repository URL when submitting a task. Asks git
// itself (`git remote get-url origin`) rather than parsing .git/config, so worktrees, includeIf and
// insteadOf rewrites are honoured exactly as the user's own git sees them.

import { execFile } from 'node:child_process'

export function gitRemoteUrl(cwd: string, remote = 'origin'): Promise<string | undefined> {
  return new Promise((resolve) => {
    execFile('git', ['remote', 'get-url', remote], { cwd, timeout: 5000 }, (error, stdout) => {
      if (error) return resolve(undefined) // not a git repo, no such remote, or git not installed
      const url = stdout.trim()
      resolve(url || undefined)
    })
  })
}

/**
 * `git@host:org/repo.git` → `https://host/org/repo` so the default matches the clone URL form the
 * gateway's git host allowlist (GIT_ALLOWED_HOSTS) and the sandbox's HTTPS clone expect. An
 * http(s) URL is returned unchanged apart from a trailing `.git`; anything else is left alone —
 * the user edits the field before submitting either way.
 */
export function normaliseRemote(url: string): string {
  const scp = /^(?:[\w.-]+@)?([\w.-]+):(?!\/\/)(.+?)(?:\.git)?\/?$/.exec(url)
  if (scp && !/^[a-z][a-z0-9+.-]*:\/\//i.test(url)) return `https://${scp[1]}/${scp[2]}`
  if (/^https?:\/\//i.test(url)) return url.replace(/\.git\/?$/, '')
  return url
}
