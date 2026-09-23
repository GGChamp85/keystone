import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { gitRemoteUrl, normaliseRemote } from '../git'
import { countActive } from '../tasks'

describe('normaliseRemote', () => {
  it('turns an scp-style remote into the https clone URL', () => {
    expect(normaliseRemote('git@git.example.internal:acme/greeter.git')).toBe('https://git.example.internal/acme/greeter')
    expect(normaliseRemote('gitea@gitea.internal.keystone.local:team/api')).toBe('https://gitea.internal.keystone.local/team/api')
  })
  it('strips a trailing .git from http(s) and leaves other schemes alone', () => {
    expect(normaliseRemote('https://git.example.internal/acme/greeter.git')).toBe('https://git.example.internal/acme/greeter')
    expect(normaliseRemote('http://localhost:3000/acme/greeter')).toBe('http://localhost:3000/acme/greeter')
    expect(normaliseRemote('ssh://git@host/acme/greeter.git')).toBe('ssh://git@host/acme/greeter.git')
  })
})

describe('gitRemoteUrl (real git)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'keystone-ext-git-'))
  afterAll(() => rmSync(dir, { recursive: true, force: true }))

  it('reads origin from a real repository and returns undefined where there is none', async () => {
    expect(await gitRemoteUrl(dir)).toBeUndefined() // not a repository
    execFileSync('git', ['init', '-q'], { cwd: dir })
    expect(await gitRemoteUrl(dir)).toBeUndefined() // no origin yet
    execFileSync('git', ['remote', 'add', 'origin', 'https://git.example.internal/acme/greeter.git'], { cwd: dir })
    expect(await gitRemoteUrl(dir)).toBe('https://git.example.internal/acme/greeter.git')
  })
})

describe('countActive', () => {
  it('counts pending and running only (src/db/models.py TaskStatus)', () => {
    const tasks = ['pending', 'running', 'completed', 'failed', 'cancelled', 'timed_out', 'running'].map((status) => ({ status }))
    expect(countActive(tasks)).toBe(3)
  })
})
