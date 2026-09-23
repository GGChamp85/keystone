// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Task-status helpers with no VS Code dependency, so they can be unit-tested outside the host.

/** The statuses that count as "running" in the status bar (src/db/models.py's TaskStatus). */
export const ACTIVE_STATUSES = new Set(['pending', 'running'])

export function countActive(tasks: { status: string }[]): number {
  return tasks.filter((t) => ACTIVE_STATUSES.has(t.status)).length
}
