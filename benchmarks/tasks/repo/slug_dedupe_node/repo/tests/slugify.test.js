import test from "node:test";
import assert from "node:assert/strict";
import { slugify } from "../slug.js";

test("slugify lowercases, replaces runs of punctuation and trims dashes", () => {
  assert.equal(slugify("  Hello, World!  "), "hello-world");
  assert.equal(slugify("Ünïcode & symbols"), "n-code-symbols");
});
