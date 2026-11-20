import { test } from "node:test";
import assert from "node:assert/strict";
import { modeFromAllowedKeys, parseResearchScope, sessionScope } from "../lib/research-scope";

const a = "a".repeat(64);
const b = "b".repeat(64);

test("automatic research is the implicit safe mode for empty legacy requests", () => {
  assert.deepEqual(parseResearchScope({}), { research_mode: "automatic", allowed_version_keys: [] });
  assert.deepEqual(sessionScope("automatic", []), parseResearchScope({}));
  assert.equal(modeFromAllowedKeys([]), "automatic");
});

test("browser-selected papers cannot silently disable autonomous discovery", () => {
  assert.throws(() => parseResearchScope({ allowed_version_keys: [a] }), /explicit mode/);
  assert.throws(() => parseResearchScope({ research_mode: "automatic", allowed_version_keys: [a] }), /cannot carry/);
  assert.throws(() => parseResearchScope({ research_mode: "restricted", allowed_version_keys: [] }), /requires/);
  assert.throws(() => parseResearchScope({ research_mode: "unknown", allowed_version_keys: [] }), /Unrecognized/);
});

test("restricted scope requires 1–25 unique, canonical paper keys and is immutable in the payload", () => {
  const selected = [a, b];
  const scope = parseResearchScope({ research_mode: "restricted", allowed_version_keys: selected });
  assert.deepEqual(scope, { research_mode: "restricted", allowed_version_keys: selected });
  assert.equal(modeFromAllowedKeys(scope.allowed_version_keys), "restricted");
  selected.pop();
  assert.equal(scope.allowed_version_keys.length, 2);
  assert.throws(() => sessionScope("restricted", [a, a]), /unique/);
  assert.throws(() => sessionScope("restricted", ["not-a-key"]), /canonical/);
  assert.throws(() => sessionScope("restricted", Array(26).fill(a)), /1–25/);
});
