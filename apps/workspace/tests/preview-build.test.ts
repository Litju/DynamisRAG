import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

test("deployment build guard refuses production and unknown environments", () => {
  for (const value of ["production", "", "preview"]) {
    const result = spawnSync(
      process.execPath,
      ["scripts/require-preview.mjs"],
      {
        env: { ...process.env, VERCEL_ENV: value },
        encoding: "utf8",
      },
    );
    assert.equal(result.status, value === "preview" ? 0 : 1);
  }
});
