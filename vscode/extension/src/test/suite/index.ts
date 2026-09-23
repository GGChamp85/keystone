// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// The mocha entry point @vscode/test-electron loads inside the VS Code extension host.
import * as fs from 'node:fs'
import * as path from 'node:path'
import Mocha from 'mocha'

export function run(): Promise<void> {
  const mocha = new Mocha({ ui: 'tdd', color: true, timeout: 120_000 })
  const dir = __dirname
  for (const file of fs.readdirSync(dir)) {
    if (file.endsWith('.test.js')) mocha.addFile(path.join(dir, file))
  }
  return new Promise((resolve, reject) => {
    mocha.run((failures) => (failures > 0 ? reject(new Error(`${failures} test(s) failed`)) : resolve()))
  })
}
