import { test, expect } from "@playwright/test";

test("authenticated Eve conversation executes the Python graph and denies a second user", async ({
  page,
  browser,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Document library" }),
  ).toBeVisible();
  await expect(page.locator(".paper")).toHaveCount(
    process.env.WORKSPACE_LICENSED_CORPUS ? 3 : 2,
  );
  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill("What happened to Achilles tendon stiffness during depth jumping?");
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toBeVisible({ timeout: 90000 });
  await page
    .getByRole("button", { name: "Inspect source 1", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Canonical source inspection" }),
  ).toContainText("source_spans");
  await expect(page.locator("mark")).toBeVisible();
  await page.screenshot({
    path: "test-results/source-inspection.png",
    fullPage: true,
  });
  const threads = await (
    await page.request.get("/api/research/threads")
  ).json();
  const current = threads.find(
    (t: { external_ref: string | null }) =>
      t.external_ref && t.external_ref !== "workspace-catalog",
  );
  const bob = await browser.newContext({
    httpCredentials: { username: "bob", password: "workspace-test-only" },
  });
  const denied = await bob.request.get(
    `http://127.0.0.1:3018/api/conversation/eve/v1/session/${current.external_ref}/stream`,
  );
  expect(denied.status()).toBe(403);
  const deniedThread = await bob.request.get(
    `http://127.0.0.1:3018/api/research/threads/${current.thread_id}`,
  );
  expect(deniedThread.status()).toBe(403);
  await bob.close();
  await page.getByRole("button", { name: "Close source" }).click();
  await page
    .getByLabel("Ask a research question")
    .fill("What about its measurement protocol?");
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(page.locator(".message.user")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Research ↑" })).toBeVisible({
    timeout: 90000,
  });
  await page.reload();
  await expect(page.locator(".message.user")).toHaveCount(2, {
    timeout: 30000,
  });
});

test("mobile workspace fits and rejects unauthenticated API requests", async ({
  page,
  playwright,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const anonymous = await playwright.request.newContext({
    httpCredentials: { username: "", password: "" },
  });
  expect(
    (
      await anonymous.get("http://127.0.0.1:3018/api/research/threads")
    ).status(),
  ).toBe(401);
  await anonymous.dispose();
});

test("cross-paper comparison renders the verified study matrix", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill(
      "Compare the methodology and results of the two plyometric training studies",
    );
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(page.getByRole("table")).toBeVisible({ timeout: 90000 });
  expect(
    await page.getByRole("table").getByRole("row").count(),
  ).toBeGreaterThanOrEqual(3);
});

test("document selection is immutable and unsupported questions abstain", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".paper")).toHaveCount(
    process.env.WORKSPACE_LICENSED_CORPUS ? 3 : 2,
  );
  await page.locator(".paper input").first().check();
  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.locator(".paper input:checked")).toHaveCount(1);
  await expect(page.locator(".paper input").first()).toBeDisabled();
  await page
    .getByLabel("Ask a research question")
    .fill("What is the orbital period of the exoplanet Kepler-22b?");
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(page.locator(".results")).toContainText("abstained", {
    timeout: 90000,
  });
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toHaveCount(0);
});

test("cancellation reaches the Python run and the conversation accepts a follow-up", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  const response = page.waitForResponse((r) =>
    /\/api\/research\/runs\/run-[^/]+$/.test(r.url()),
  );
  await page
    .getByLabel("Ask a research question")
    .fill(
      "[browser-slow] What happened to Achilles tendon stiffness during depth jumping?",
    );
  await page.getByRole("button", { name: "Research ↑" }).click();
  const runUrl = (await response).url();
  await page.getByRole("button", { name: "Cancel research" }).click();
  await expect
    .poll(async () => (await (await page.request.get(runUrl)).json()).status, {
      timeout: 30000,
    })
    .toBe("cancelled");
  await expect(page.getByRole("button", { name: "Research ↑" })).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill("What happened to Achilles tendon stiffness during depth jumping?");
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toBeVisible({ timeout: 90000 });
});

test("worker failures are explicit and a later turn recovers", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill(
      "[browser-failure] What happened to Achilles tendon stiffness during depth jumping?",
    );
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(page.locator(".results")).toContainText("failed", {
    timeout: 90000,
  });
  await expect(page.getByRole("button", { name: "Research ↑" })).toBeVisible();
  await page
    .getByLabel("Ask a research question")
    .fill("What happened to Achilles tendon stiffness during depth jumping?");
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toBeVisible({ timeout: 90000 });
});
