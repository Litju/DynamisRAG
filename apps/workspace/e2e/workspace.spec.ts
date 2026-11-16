import { test, expect } from "@playwright/test";

const testOrigin = process.env.WORKSPACE_TEST_URL || "http://127.0.0.1:3018";

test.beforeEach(async ({ page }, info) => {
  if (info.title.startsWith("a visitor signs in")) return;
  await page.goto("/login");
  await page.getByLabel("Username", { exact: true }).fill("alice");
  await page.getByLabel("Password", { exact: true }).fill(process.env.WORKSPACE_TEST_PASSWORD || "workspace-test-only");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Document library" })).toBeVisible();
});

test("a visitor signs in with the form and asks a cited question", async ({ browser }) => {
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.goto(testOrigin);
    await expect(page.getByRole("heading", { name: "Sign in to DynamisRAG" })).toBeVisible();
    await page.getByLabel("Username", { exact: true }).fill("alice");
    await page.getByLabel("Password", { exact: true }).fill("incorrect-password");
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("Username or password is incorrect");
    await page.getByLabel("Username", { exact: true }).fill("alice");
    await page.getByLabel("Password", { exact: true }).fill(process.env.WORKSPACE_TEST_PASSWORD || "workspace-test-only");
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Document library" })).toBeVisible();
    await page.getByRole("button", { name: "Start a research thread" }).click();
    await expect(page.getByLabel("Ask a research question")).toBeVisible({ timeout: 30000 });
    await page.getByLabel("Ask a research question").fill("What happened to Achilles tendon stiffness during depth jumping?");
    await page.getByRole("button", { name: "Research ↑" }).click();
    await expect(page.getByRole("button", { name: "Inspect source 1", exact: true })).toBeVisible({ timeout: 90000 });
  } finally { await context.close(); }
});

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
  const firstRun = page.waitForResponse((r) =>
    /\/api\/research\/runs\/run-[^/]+$/.test(r.url()),
  );
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
  const runUrl = (await firstRun).url();
  const events = await (await page.request.get(`${runUrl}/events`)).text();
  const ids = [...events.matchAll(/^id: (\d+)$/gm)].map((match) =>
    Number(match[1]),
  );
  expect(ids.length).toBeGreaterThan(1);
  expect([...ids].sort((a, b) => a - b)).toEqual(ids);
  const replay = await (
    await page.request.get(`${runUrl}/events`, {
      headers: { "Last-Event-ID": String(ids.at(-2)) },
    })
  ).text();
  expect(
    [...replay.matchAll(/^id: (\d+)$/gm)].map((match) => Number(match[1])),
  ).toEqual([ids.at(-1)]);
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
    httpCredentials: { username: "bob", password: process.env.WORKSPACE_TEST_BOB_PASSWORD || "workspace-test-only" },
  });
  for (const ref of [current.external_ref.slice(4), current.external_ref]) {
    const claim = await bob.request.post(
      `${testOrigin}/api/research/threads`,
      {
        data: { external_ref: ref },
        headers: { origin: testOrigin },
      },
    );
    expect(claim.status()).toBe(400);
  }
  const denied = await bob.request.get(
    `${testOrigin}/api/conversation/eve/v1/session/${current.external_ref.slice(4)}/stream`,
  );
  expect(denied.status()).toBe(403);
  expect(
    (
      await bob.request.post(
        `${testOrigin}/api/conversation/eve/v1/session/${current.external_ref.slice(4)}/cancel`,
        { headers: { origin: testOrigin } },
      )
    ).status(),
  ).toBe(403);
  expect(
    (
      await bob.request.post(
        `${testOrigin}/api/conversation/eve/v1/session/${current.external_ref.slice(4)}`,
        {
          data: { message: "forged follow-up" },
          headers: { origin: testOrigin },
        },
      )
    ).status(),
  ).toBe(403);
  const deniedThread = await bob.request.get(
    `${testOrigin}/api/research/threads/${current.thread_id}`,
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
      await anonymous.get(`${testOrigin}/api/research/threads`)
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

test("the sample preview explains its limits for force velocity profiling", async ({ page }) => {
  test.skip(!process.env.WORKSPACE_LICENSED_CORPUS, "Uses the licensed sample corpus.");
  await page.goto("/");
  await expect(page.getByRole("note").first()).toContainText("Europe PMC and OpenAlex");
  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({ timeout: 30000 });
  await page.getByLabel("Ask a research question").fill("lets research about force velocity profiling in jumps, need a method using LPT and Force plates");
  await page.getByRole("button", { name: "Research ↑" }).click();
  await expect(page.getByRole("heading", { name: "No synthesized answer" })).toBeVisible({ timeout: 90000 });
  await expect(page.locator(".results")).toContainText("not a finding that the method or evidence does not exist");
  await expect(page.getByRole("button", { name: "Inspect source 1", exact: true })).toHaveCount(0);
});

test("cancellation reaches the Python run and the conversation accepts a follow-up", async ({
  page,
}) => {
  test.skip(!!process.env.WORKSPACE_LICENSED_CORPUS,
    "The slow marker belongs to the synthetic backend; licensed cancellation requires a queued run.");
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
