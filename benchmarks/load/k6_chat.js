// Copyright 2024-2026 Gaurav Gupta / Keystone
// Licensed under the Apache License, Version 2.0
//
// Load test for the gateway's chat completions: real requests, real token usage, real latency
// distribution — the numbers docs/benchmarks/load-test.md publishes come from this script against a
// GPU deployment. CI runs it as a smoke (few users, short) against the CPU demo model to keep the
// script itself honest; the smoke's numbers say nothing about GPU throughput.
//
//   k6 run -e KEYSTONE_URL=http://localhost:8080 -e KEYSTONE_API_KEY=ks-... benchmarks/load/k6_chat.js
//   k6 run -e VUS=20 -e DURATION=5m -e MODEL=coding -e MAX_TOKENS=256 ... benchmarks/load/k6_chat.js
//
import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter } from 'k6/metrics';

const BASE = __ENV.KEYSTONE_URL || 'http://localhost:8080';
const KEY = __ENV.KEYSTONE_API_KEY;
const MODEL = __ENV.MODEL || 'coding';
const MAX_TOKENS = Number(__ENV.MAX_TOKENS || 64);
const STREAM = (__ENV.STREAM || '0') === '1';

export const options = {
  vus: Number(__ENV.VUS || 2),
  duration: __ENV.DURATION || '20s',
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  thresholds: {
    // the smoke asserts the plumbing; a GPU run tightens these from its own measurements
    'checks': ['rate>0.99'],
    'http_req_failed': ['rate<0.01'],
  },
};

const promptTokens = new Trend('keystone_prompt_tokens');
const completionTokens = new Trend('keystone_completion_tokens');
const tokensPerSecond = new Trend('keystone_completion_tokens_per_second');
const routedFallback = new Counter('keystone_routed_to_fallback');

const PROMPTS = [
  'Write a Python function that reverses a string, with a docstring.',
  'Explain in two sentences what a token bucket rate limiter does.',
  'Write a SQL query that returns the ten most recent orders per customer.',
  'Given a list of integers, return the second largest. Python, no imports.',
];

export default function () {
  const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
  const body = JSON.stringify({
    model: MODEL,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: MAX_TOKENS,
    temperature: 0.2,
    stream: STREAM,
    ...(STREAM ? { stream_options: { include_usage: true } } : {}),
  });
  const started = Date.now();
  const res = http.post(`${BASE}/v1/chat/completions`, body, {
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
    timeout: '120s',
  });
  const elapsed = (Date.now() - started) / 1000;

  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'has X-VS-Route-Decision': (r) => !!r.headers['X-Vs-Route-Decision'],
  });
  if (!ok) return;
  if ((res.headers['X-Vs-Route-Decision'] || '').includes('reason=fallback')) routedFallback.add(1);

  let usage = null;
  if (STREAM) {
    for (const line of res.body.split('\n')) {
      if (!line.startsWith('data: ') || line.includes('[DONE]')) continue;
      try {
        const chunk = JSON.parse(line.slice(6));
        if (chunk.usage) usage = chunk.usage;
      } catch (_) {
        /* keep-alive or partial frame */
      }
    }
  } else {
    try {
      usage = JSON.parse(res.body).usage;
    } catch (_) {
      usage = null;
    }
  }
  check(usage, { 'real usage reported': (u) => !!u && u.completion_tokens > 0 });
  if (usage) {
    promptTokens.add(usage.prompt_tokens);
    completionTokens.add(usage.completion_tokens);
    if (elapsed > 0) tokensPerSecond.add(usage.completion_tokens / elapsed);
  }
}

export function handleSummary(data) {
  const out = __ENV.SUMMARY_JSON;
  const summary = {
    model: MODEL,
    vus: options.vus,
    duration: options.duration,
    stream: STREAM,
    requests: data.metrics.http_reqs ? data.metrics.http_reqs.values.count : 0,
    failed_rate: data.metrics.http_req_failed ? data.metrics.http_req_failed.values.rate : null,
    latency_ms: data.metrics.http_req_duration
      ? {
          p50: data.metrics.http_req_duration.values.med,
          p95: data.metrics.http_req_duration.values['p(95)'],
          p99: data.metrics.http_req_duration.values['p(99)'],
        }
      : null,
    completion_tokens_per_second: data.metrics.keystone_completion_tokens_per_second
      ? data.metrics.keystone_completion_tokens_per_second.values
      : null,
    routed_to_fallback: data.metrics.keystone_routed_to_fallback ? data.metrics.keystone_routed_to_fallback.values.count : 0,
  };
  const result = { stdout: JSON.stringify(summary, null, 2) + '\n' };
  if (out) result[out] = JSON.stringify(summary, null, 2);
  return result;
}
