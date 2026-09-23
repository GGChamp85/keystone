import test from "node:test";
import assert from "node:assert/strict";
import { retry } from "../backoff.js";

test("retry returns the first successful result and rethrows the last error", async () => {
  let calls = 0;
  const value = await retry(async () => {
    calls += 1;
    if (calls < 3) throw new Error("not yet");
    return "done";
  }, { attempts: 3, base: 1, cap: 2 });
  assert.equal(value, "done");
  assert.equal(calls, 3);
  await assert.rejects(retry(async () => { throw new Error("always"); }, { attempts: 2, base: 1, cap: 2 }), /always/);
});
