// Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
//
// Bundles src/extension.ts into dist/extension.js — one CommonJS file the extension host loads,
// `vscode` left external (it is provided by the host). `node esbuild.mjs --watch` rebuilds on save.
import * as esbuild from 'esbuild'

const watch = process.argv.includes('--watch')

const options = {
  entryPoints: ['src/extension.ts'],
  bundle: true,
  outfile: 'dist/extension.js',
  platform: 'node',
  format: 'cjs',
  target: 'node20', // VS Code 1.90's Electron ships Node 20 — global fetch and ReadableStream are available
  external: ['vscode'],
  sourcemap: true,
  minify: false,
  logLevel: 'info',
}

if (watch) {
  const ctx = await esbuild.context(options)
  await ctx.watch()
} else {
  await esbuild.build(options)
}
