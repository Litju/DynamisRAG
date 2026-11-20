import { test, expect, type Page } from "@playwright/test";

const testOrigin = process.env.WORKSPACE_TEST_URL || "http://127.0.0.1:3018";

async function latestRunUrl(page: Page) {
  let runId: string | undefined;
  await expect.poll(async () => {
    const thread = new URL(page.url()).searchParams.get("thread");
    if (!thread) return undefined;
    const response = await page.request.get("/api/research/threads/" + thread + "/runs");
    if (!response.ok()) return undefined;
    runId = (await response.json())[0]?.run_id;
    return runId;
  }, { timeout: 15_000 }).toBeTruthy();
  return testOrigin + "/api/research/runs/" + runId;
}

test.beforeEach(async ({ page }, info) => {
  if (info.title.startsWith("a visitor signs in")) return;
  await page.goto("/login");
  await page.getByLabel("Username", { exact: true }).fill("alice");
  await page.getByLabel("Password", { exact: true }).fill(process.env.WORKSPACE_TEST_PASSWORD || "workspace-test-only");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Document library" })).toBeVisible();
  await expect(page.locator(".paper").first()).toBeVisible();
});

test("a visitor signs in with the form and asks a cited question", async ({ page }) => {
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
  await expect(page.locator(".paper").first()).toBeVisible();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({ timeout: 30000 });
  await page.getByLabel("Ask a research question").fill("What happened to Achilles tendon stiffness during depth jumping?");
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(page.getByRole("button", { name: "Inspect source 1", exact: true })).toBeVisible({ timeout: 90000 });
  const finding = page.locator(".findings li").first();
  // The verified quotation leads; any model wording is labelled and its recorded check shown.
  await expect(finding.locator(".finding-quote")).toContainText("“");
  await expect(finding.locator(".finding-checks")).toContainText("Wording check:");
  const paraphrase = finding.locator(".finding-paraphrase");
  if (await paraphrase.count()) await expect(paraphrase).toContainText("not independently verified");
  // A finished conversation stops polling, so an idle tab cannot renew the session.
  await page.waitForTimeout(1_500);
  let polls = 0;
  page.on("request", (request) => { if (/\/api\/research\/threads\/[^/]+\/runs$/.test(new URL(request.url()).pathname)) polls += 1; });
  await page.waitForTimeout(9_000);
  expect(polls).toBe(0);
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
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill("What happened to Achilles tendon stiffness during depth jumping?");

  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toBeVisible({ timeout: 90000 });
  await page
    .getByRole("button", { name: "Inspect source 1", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "Canonical source inspection" }),
  ).toContainText("Source coordinates and surrounding context");
  await expect(page.locator("mark")).toBeVisible();
  const runUrl = await latestRunUrl(page);
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
    (t: { thread_id: string; external_ref: string | null }) =>
      t.external_ref && new URL(page.url()).searchParams.get("thread") === t.thread_id,
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
  await page.getByRole("button", { name: "Close canonical source inspection" }).click();
  await page
    .getByLabel("Ask a research question")
    .fill("What about its measurement protocol?");
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(page.getByLabel("Evidence for turn").locator("option")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Start research", exact: true })).toBeVisible({
    timeout: 90000,
  });
  await page.reload();
  await expect(page.getByLabel("Evidence for turn").locator("option")).toHaveCount(2, {
    timeout: 30000,
  });
  await page.getByLabel("Evidence for turn").selectOption({ label: "Turn 1" });
  await page.getByRole("button", { name: "Inspect source 1", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Canonical source inspection" })).toBeVisible();
  await expect(page.locator("mark")).toContainText(/stiffness/i);
});

test("mobile workspace fits and rejects unauthenticated API requests", async ({
  page,
  playwright,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
  for (const width of [390, 820, 1440]) {
    await page.setViewportSize({ width, height: 844 });
    await expect(page.getByLabel("Ask a research question")).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (width < 1090) {
      await page.getByRole("button", { name: "Library", exact: true }).click();
      await expect(page.getByRole("dialog", { name: "Source library" })).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.getByRole("button", { name: "Library", exact: true })).toBeFocused();
    }
  }
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

test("paper browsing cannot silently restrict automatic research", async ({ page }) => {
  await expect(page.getByRole("radio", { name: /Automatic research/ })).toBeChecked();
  const stack = page.getByRole("region", { name: "Browse saved papers" });
  await expect(stack.getByRole("button", { name: "Add to scope" })).toHaveCount(0);
  await expect(page.locator(".paper input:checked")).toHaveCount(0);
  await page.getByRole("radio", { name: /Restrict to selected papers/ }).check();
  const first = await stack.locator(".stack-paper p").innerText();
  await stack.getByRole("button", { name: "Next paper" }).click();
  await expect(stack.locator(".stack-paper p")).not.toHaveText(first);
  await stack.getByRole("button", { name: "Add to scope" }).click();
  await expect(page.locator(".paper input:checked")).toHaveCount(1);
  await stack.getByRole("button", { name: "In scope" }).click();
  await expect(page.locator(".paper input:checked")).toHaveCount(0);
  await stack.getByRole("button", { name: "Pause motion" }).click();
  await expect(stack.getByRole("button", { name: "Resume motion" })).toHaveAttribute("aria-pressed", "true");
});

test("cross-paper comparison renders the verified study matrix", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill(
      "Compare the methodology and results of the two plyometric training studies",
    );
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  const comparison = page.locator(".study-comparison");
  await expect(comparison).toBeVisible({ timeout: 90000 });
  await comparison.getByText("Study methods and reported results").click();
  expect(await comparison.locator(".study").count()).toBeGreaterThanOrEqual(2);
});

test("document selection is immutable and unsupported questions abstain", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".paper")).toHaveCount(
    process.env.WORKSPACE_LICENSED_CORPUS ? 3 : 2,
  );
  await page.getByRole("radio", { name: /Restrict to selected papers/ }).check();
  await page.locator(".paper input").first().check();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.locator(".paper input:checked")).toHaveCount(1);
  await page
    .getByLabel("Ask a research question")
    .fill("What is the orbital period of the exoplanet Kepler-22b?");
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(page.locator(".paper input").first()).toBeDisabled();
  await expect(page.locator(".result")).toContainText("Evidence was insufficient", {
    timeout: 90000,
  });
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toHaveCount(0);
});

test("the sample preview explains its limits for force velocity profiling", async ({ page }) => {
  test.skip(!process.env.WORKSPACE_LICENSED_CORPUS, "Uses the licensed sample corpus.");
  await page.goto("/");
  await expect(page.getByLabel("Ask a research question")).toBeVisible({ timeout: 30000 });
  await page.getByLabel("Ask a research question").fill("lets research about force velocity profiling in jumps, need a method using LPT and Force plates");
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Evidence was insufficient" })).toBeVisible({ timeout: 90000 });
  await expect(page.locator(".result")).toContainText("within the selected scope");
  await expect(page.getByRole("button", { name: "Inspect source 1", exact: true })).toHaveCount(0);
});

test("cancellation reaches the Python run and the conversation accepts a follow-up", async ({
  page,
}) => {
  test.skip(!!process.env.WORKSPACE_LICENSED_CORPUS,
    "The slow marker belongs to the synthetic backend; licensed cancellation requires a queued run.");
  await page.goto("/");
  let feedRequests = 0;
  await page.route("**/api/research/runs/*/events?*", async (route) => {
    if (++feedRequests === 1) await route.fulfill({ status: 503 });
    else await route.continue();
  });
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });

  await page
    .getByLabel("Ask a research question")
    .fill(
      "[browser-slow] What happened to Achilles tendon stiffness during depth jumping?",
    );
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  const runUrl = await latestRunUrl(page);
  await expect.poll(() => feedRequests, { timeout: 10_000, intervals: [100, 250, 500] }).toBeGreaterThan(1);
  await expect(page.getByText(/Activity feed reconnecting/)).toBeHidden();

  await page.getByRole("button", { name: "Cancel research" }).click();
  await expect
    .poll(async () => (await (await page.request.get(runUrl)).json()).status, {
      timeout: 30000,
    })
    .toBe("cancelled");
  await expect(page.getByRole("button", { name: "Start research", exact: true })).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill("What happened to Achilles tendon stiffness during depth jumping?");
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toBeVisible({ timeout: 90000 });
});

test("worker failures are explicit and a later turn recovers", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByLabel("Ask a research question")).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByLabel("Ask a research question")
    .fill(
      "[browser-failure] What happened to Achilles tendon stiffness during depth jumping?",
    );
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(page.locator(".result")).toContainText("Research could not finish", {
    timeout: 90000,
  });
  await expect(page.getByRole("button", { name: "Start research", exact: true })).toBeVisible();
  await page
    .getByLabel("Ask a research question")
    .fill("What happened to Achilles tendon stiffness during depth jumping?");
  await page.getByRole("button", { name: "Start research", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Inspect source 1", exact: true }),
  ).toBeVisible({ timeout: 90000 });
});
