import { defineConfig } from 'vitest/config'

// Unit tests for the parts of the extension that do not need a VS Code host: the SSE decoder, the
// model picker ordering and the step renderer — run against the real recorded fixtures in
// src/__tests__/fixtures (scripts/record_task_stream_fixture.py). The VS Code host tests live in
// src/test and run under @vscode/test-electron (`npm run test:integration`).
export default defineConfig({
  test: {
    include: ['src/__tests__/**/*.test.ts'],
    environment: 'node',
  },
})
