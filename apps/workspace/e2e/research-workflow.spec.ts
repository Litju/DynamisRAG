import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { test, expect, type Page } from "@playwright/test";

// Live acceptance against the real Python API/worker, real Europe PMC/OpenAlex, a fresh empty
// canonical PostgreSQL, fresh Qdrant and the pinned TEI. Opt-in only; never runs in CI.
const enabled = !!process.env.WORKSPACE_RESEARCH_WORKFLOW;
const providerBlocked = process.env.WORKSPACE_PROVIDER_BLOCKED === "1";
const evidenceDir = process.env.WORKSPACE_RESEARCH_EVIDENCE_DIR || "test-results/research-workflow";

const INITIAL =
  "What does the evidence say about the validity, reliability, practical applications and " +
  "limitations of force–velocity profiling using linear position transducers (LPT) in " +
  "resistance training?";
const FOLLOW_UP =
  "How do LPT-derived force–velocity measures compare with reference systems such as force " +
  "plates or 3D motion capture?";

function save(name: string, value: unknown) {
  mkdirSync(evidenceDir, { recursive: true });
  writeFileSync(join(evidenceDir, name), JSON.stringify(value, null, 1) + "\n");
}

async function ask(page: Page, question: string, index: number) {
  const runResponse = page.waitForResponse(
    (r) => /\/api\/research\/runs\/run-[^/?]+$/.test(r.url()) && r.request().method() === "GET",
    { timeout: 120000 },
  );
  await page.getByLabel("Ask a research question").fill(question);
  await page.getByRole("button", { name: "Research ↑" }).click();
  const runUrl = (await runResponse).url();
  await expect
    .poll(async () => (await (await page.request.get(runUrl)).json()).status, {
      timeout: 1_500_000,
      intervals: [5000],
    })
    .toMatch(/^(completed|abstained|failed|cancelled)$/);
  const run = await (await page.request.get(runUrl)).json();
  const events = await (await page.request.get(`${runUrl}/events`)).text();
  save(`turn${index}-run.json`, run);
  writeFileSync(join(evidenceDir, `turn${index}-events.sse`), events);
  await expect(page.locator(".results h2")).toBeVisible({ timeout: 60000 });
  return { run, events };
}

test.describe.configure({ timeout: 3_600_000 });

test("live research workflow: empty corpus, real discovery, ingestion, retrieval, follow-up", async ({
  page,
}) => {
  test.skip(!enabled, "Set WORKSPACE_RESEARCH_WORKFLOW=1 against the live research stack.");
  await page.goto("/login");
  await page.getByLabel("Username", { exact: true }).fill("alice");
  await page
    .getByLabel("Password", { exact: true })
    .fill(process.env.WORKSPACE_TEST_PASSWORD || "workspace-test-only");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Document library" })).toBeVisible();
  const initialPapers = await page.locator(".paper").count();
  save("initial-library.json", { papers: initialPapers });
  if (process.env.WORKSPACE_EXPECT_EMPTY_CORPUS === "1") expect(initialPapers).toBe(0);
  await page.screenshot({ path: join(evidenceDir, "0-empty-library.png"), fullPage: true });

  await page.getByRole("button", { name: "Start a research thread" }).click();
  await expect(page.getByLabel("Ask a research question")).toBeVisible({ timeout: 30000 });

  const first = await ask(page, INITIAL, 1);
  const acquisition = first.run.telemetry?.literature_acquisition;
  expect(acquisition?.status).toBe("completed");
  expect(acquisition.ingested.length).toBeGreaterThan(0);
  expect(first.events).toContain("event: literature.discovered");
  expect(first.events).toContain("event: corpus.projected");
  await expect(page.getByTestId("ingested-paper").first()).toContainText("DOI 10.");
  await expect(page.getByTestId("qdrant-collection")).toContainText("qdrant-passages-v1-");
  await page.screenshot({ path: join(evidenceDir, "1-initial-answer.png"), fullPage: true });

  if (providerBlocked) {
    // No model may answer: the run abstains, either at synthesis (provider blocked) or at the
    // relevance gate. Retrieval over the NEW projection must still have happened.
    expect(first.run.status).toBe("abstained");
    expect(first.run.abstention_reason).toMatch(/^(provider_authorization_blocked|evidence_gate:)/);
    expect(first.run.telemetry.model_calls).toBe(0);
    expect(first.run.telemetry.retrieval_candidates.length).toBeGreaterThan(0);
    expect(first.run.telemetry.retrieval_candidates[0].retrieval_mode).toBe("qdrant_hybrid_rrf");
    await expect(page.getByTestId("retrieved-passage").first()).toBeVisible();
    await page.getByRole("button", { name: "Inspect retrieved passage" }).first().click();
  } else {
    expect(first.run.status).toBe("completed");
    expect(first.run.verified_citations.length).toBeGreaterThan(0);
    await page.getByRole("button", { name: "Inspect source 1", exact: true }).click();
    await expect(page.locator("mark")).toBeVisible();
  }
  await expect(page.getByRole("region", { name: "Canonical source inspection" })).toContainText(
    "source_spans",
  );
  await page.screenshot({ path: join(evidenceDir, "2-source-inspection.png"), fullPage: true });
  await page.getByRole("button", { name: "Close source" }).click();

  const second = await ask(page, FOLLOW_UP, 2);
  await page.screenshot({ path: join(evidenceDir, "3-follow-up.png"), fullPage: true });
  if (providerBlocked) {
    expect(second.run.status).toBe("abstained");
    expect(second.run.abstention_reason).toMatch(/^(provider_authorization_blocked|evidence_gate:)/);
    expect(second.run.telemetry.model_calls).toBe(0);
    expect(second.run.telemetry.retrieval_candidates.length).toBeGreaterThan(0);
  } else {
    expect(second.run.status).toBe("completed");
    expect(second.run.verified_citations.length).toBeGreaterThan(0);
  }
  save("acceptance.json", {
    provider_blocked: providerBlocked,
    full_acceptance: !providerBlocked,
    initial_status: first.run.status,
    initial_abstention: first.run.abstention_reason,
    follow_up_status: second.run.status,
    follow_up_abstention: second.run.abstention_reason,
  });
});
