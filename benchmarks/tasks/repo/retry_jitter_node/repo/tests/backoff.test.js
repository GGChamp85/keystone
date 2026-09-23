import test from "node:test";
import assert from "node:assert/strict";
import { backoffDelay } from "../backoff.js";

test("the delay never exceeds the cap and stays within the exponential window", () => {
  for (const r of [0, 0.25, 0.5, 0.999, 1]) {
    for (let attempt = 0; attempt < 12; attempt++) {
      const d = backoffDelay(attempt, { base: 100, cap: 5000, random: () => r });
      assert.ok(d >= 0, `attempt ${attempt} r=${r}: negative delay ${d}`);
      assert.ok(d <= 5000, `attempt ${attempt} r=${r}: ${d} exceeds the cap`);
      assert.ok(d <= Math.min(5000, 100 * 2 ** attempt) + 1e-9, `attempt ${attempt} r=${r}: ${d} exceeds the window`);
    }
  }
});
