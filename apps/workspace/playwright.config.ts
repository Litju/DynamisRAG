import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  timeout: 120000,
  workers: 1,
  use: {
    baseURL: process.env.WORKSPACE_TEST_URL || "http://127.0.0.1:3018",
    httpCredentials: { username: "alice", password: process.env.WORKSPACE_TEST_PASSWORD || "workspace-test-only" },
    trace: "retain-on-failure",
  },
});
