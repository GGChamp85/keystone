// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The model picker's ordering, as a pure function over GET /v1/keystone/models so it can be
// tested against a recorded real response (src/__tests__/fixtures/model-library.json).
//
// What a task can name as `model` is a gateway ROLE — coding | coding_fallback | reasoning — or
// `auto` (src/api/models/requests.py's AgentTaskRequest.model); adapters and catalog entries in
// the library are not submittable values (a promoted adapter is applied by the router per tenant),
// so only `kind: "role"` entries are listed. READY roles come first; a role whose endpoint is
// UNHEALTHY / NOT_DEPLOYED / TRAINING is shown with its state and cannot be picked — submitting
// to it would only fail later in the task. `auto` is always selectable.

import type { LibraryModel, ModelLibrary, ModelState } from './api'

export type PickState = ModelState | 'AUTO'

export interface ModelPickItem {
  id: string
  label: string
  description: string
  detail: string
  state: PickState
  selectable: boolean
  isDefault: boolean
}

export const ROLE_ORDER = ['coding', 'coding_fallback', 'reasoning'] as const

const STATE_BADGE: Record<PickState, string> = {
  AUTO: '$(sparkle)',
  READY: '$(pass-filled)',
  UNHEALTHY: '$(error)',
  NOT_DEPLOYED: '$(circle-slash)',
  TRAINING: '$(sync)',
}

function roleItem(m: LibraryModel, isDefault: boolean): ModelPickItem {
  const selectable = m.state === 'READY'
  const served = m.served_model_ids?.length ? ` · serving ${m.served_model_ids.join(', ')}` : ''
  const context = m.context ? ` · ${m.context.toLocaleString()} ctx` : ''
  return {
    id: m.id,
    label: `${STATE_BADGE[m.state]} ${m.id}`,
    description: `${m.state}${isDefault ? ' · default' : ''}${selectable ? '' : ' · not selectable'}`,
    detail: `${m.name}${m.purpose ? ` — ${m.purpose}` : ''}${served}${context}`,
    state: m.state,
    selectable,
    isDefault,
  }
}

export function autoItem(isDefault: boolean): ModelPickItem {
  return {
    id: 'auto',
    label: `${STATE_BADGE.AUTO} auto`,
    description: `Keystone picks the role${isDefault ? ' · default' : ''}`,
    detail: 'The router classifies the task description once, before the task is persisted; the resolved role is recorded on the task.',
    state: 'AUTO',
    selectable: true,
    isDefault,
  }
}

/**
 * Order: the configured default first when it is selectable, then READY roles in the gateway's
 * role order, then `auto` (if not already placed), then every non-READY role in role order, marked
 * not selectable. The gateway's own order (ROLE_ORDER) is kept within each group so the picker is
 * stable across refreshes.
 */
export function orderModelsForPicker(library: ModelLibrary, defaultModel: string): ModelPickItem[] {
  const roles = library.models.filter((m) => m.kind === 'role')
  const rank = (id: string): number => {
    const i = (ROLE_ORDER as readonly string[]).indexOf(id)
    return i === -1 ? ROLE_ORDER.length : i
  }
  roles.sort((a, b) => rank(a.id) - rank(b.id))

  const ready = roles.filter((m) => m.state === 'READY').map((m) => roleItem(m, m.id === defaultModel))
  const notReady = roles.filter((m) => m.state !== 'READY').map((m) => roleItem(m, m.id === defaultModel))
  const auto = autoItem(defaultModel === 'auto')

  const ordered = [...ready, auto, ...notReady]
  const defaultIndex = ordered.findIndex((item) => item.isDefault && item.selectable)
  if (defaultIndex > 0) {
    const [item] = ordered.splice(defaultIndex, 1)
    ordered.unshift(item)
  }
  return ordered
}

/** The picker when the library cannot be loaded: the fixed roles, none of them verified. */
export function fallbackPickItems(defaultModel: string): ModelPickItem[] {
  const items: ModelPickItem[] = ROLE_ORDER.map((id) => ({
    id,
    label: `$(question) ${id}`,
    description: `state unknown${id === defaultModel ? ' · default' : ''}`,
    detail: 'The model library could not be loaded — the role may or may not be up.',
    state: 'NOT_DEPLOYED',
    selectable: true,
    isDefault: id === defaultModel,
  }))
  const ordered = [autoItem(defaultModel === 'auto'), ...items]
  const defaultIndex = ordered.findIndex((item) => item.isDefault)
  if (defaultIndex > 0) {
    const [item] = ordered.splice(defaultIndex, 1)
    ordered.unshift(item)
  }
  return ordered
}
