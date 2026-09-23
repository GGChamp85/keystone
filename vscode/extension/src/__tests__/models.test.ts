// The model picker ordering against a real GET /v1/keystone/models response
// (fixtures/model-library.json — recorded through the real route with the fake vLLM fixture
// serving `coding` and dead ports behind the other roles; scripts/record_task_stream_fixture.py).
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { ModelLibrary } from '../api'
import { fallbackPickItems, orderModelsForPicker } from '../models'

const LIBRARY = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'model-library.json'), 'utf8')) as ModelLibrary

describe('the recorded library', () => {
  it('is the real shape: roles with live states, plus catalog entries', () => {
    const roles = LIBRARY.models.filter((m) => m.kind === 'role')
    expect(roles.map((m) => [m.id, m.state])).toEqual([
      ['coding', 'READY'],
      ['coding_fallback', 'UNHEALTHY'],
      ['reasoning', 'UNHEALTHY'],
    ])
    expect(LIBRARY.models.some((m) => m.kind === 'catalog')).toBe(true)
    expect(typeof LIBRARY.generated_at).toBe('number')
  })
})

describe('orderModelsForPicker', () => {
  it('lists only submittable values: auto and the roles — never adapters or catalog entries', () => {
    const ids = orderModelsForPicker(LIBRARY, 'auto').map((i) => i.id)
    expect(ids).toEqual(['auto', 'coding', 'coding_fallback', 'reasoning'])
  })

  it('puts READY roles before auto and the unhealthy ones last, marked not selectable', () => {
    const items = orderModelsForPicker(LIBRARY, 'coding')
    expect(items.map((i) => [i.id, i.state, i.selectable])).toEqual([
      ['coding', 'READY', true],
      ['auto', 'AUTO', true],
      ['coding_fallback', 'UNHEALTHY', false],
      ['reasoning', 'UNHEALTHY', false],
    ])
    expect(items[2].description).toContain('not selectable')
    expect(items[2].label).toContain('$(error)')
    expect(items[0].label).toContain('$(pass-filled)')
  })

  it('moves the configured default to the top when it is selectable', () => {
    expect(orderModelsForPicker(LIBRARY, 'auto')[0]).toMatchObject({ id: 'auto', isDefault: true })
    expect(orderModelsForPicker(LIBRARY, 'coding')[0]).toMatchObject({ id: 'coding', isDefault: true })
    expect(orderModelsForPicker(LIBRARY, 'coding')[0].description).toContain('default')
  })

  it('does not promote a default that is not READY — it stays in the not-ready group, unselectable', () => {
    const items = orderModelsForPicker(LIBRARY, 'reasoning')
    expect(items.map((i) => i.id)).toEqual(['coding', 'auto', 'coding_fallback', 'reasoning'])
    expect(items[3]).toMatchObject({ isDefault: true, selectable: false })
  })

  it('keeps the gateway role order within each group', () => {
    const shuffled: ModelLibrary = { ...LIBRARY, models: [...LIBRARY.models].reverse() }
    expect(orderModelsForPicker(shuffled, 'auto').map((i) => i.id)).toEqual(['auto', 'coding', 'coding_fallback', 'reasoning'])
  })

  it('shows what the endpoint really serves', () => {
    const coding = orderModelsForPicker(LIBRARY, 'auto').find((i) => i.id === 'coding')!
    const served = LIBRARY.models.find((m) => m.id === 'coding')!.served_model_ids ?? []
    expect(served.length).toBeGreaterThan(0)
    expect(coding.detail).toContain(`serving ${served.join(', ')}`)
  })
})

describe('fallbackPickItems (library unreachable)', () => {
  it('offers auto and the three fixed roles, all selectable but flagged unknown, default first', () => {
    const items = fallbackPickItems('coding_fallback')
    expect(items.map((i) => i.id)).toEqual(['coding_fallback', 'auto', 'coding', 'reasoning'])
    expect(items.every((i) => i.selectable)).toBe(true)
    expect(items[2].description).toContain('state unknown')
  })
})
