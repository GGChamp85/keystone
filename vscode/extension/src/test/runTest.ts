// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Launches a real VS Code (downloaded by @vscode/test-electron into .vscode-test/) with this
// extension loaded from the repository, and runs src/test/suite inside it. `npm run test:integration`.
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import { runTests } from '@vscode/test-electron'

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, '../..')
  const extensionTestsPath = path.resolve(__dirname, './suite/index')
  // A short user-data-dir: VS Code binds a Unix socket under it, and macOS/Linux cap socket paths
  // at ~103 bytes — the default (.vscode-test/user-data under a deep checkout) fails with EINVAL.
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ks-vsc-'))
  try {
    await runTests({
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: ['--disable-extensions', '--disable-workspace-trust', `--user-data-dir=${userDataDir}`],
    })
  } finally {
    fs.rmSync(userDataDir, { recursive: true, force: true })
  }
}

main().catch((err) => {
  console.error('Failed to run the VS Code integration tests', err)
  process.exit(1)
})
