import test from "node:test";
import assert from "node:assert/strict";
import { SlugRegistry } from "../slug.js";

test("a repeated title gets an incrementing suffix", () => {
  const r = new SlugRegistry();
  assert.equal(r.claim("Hello World"), "hello-world");
  assert.equal(r.claim("Hello World"), "hello-world-2");
  assert.equal(r.claim("hello world!"), "hello-world-3");
  assert.equal(r.claim("Hello World"), "hello-world-4");
});
