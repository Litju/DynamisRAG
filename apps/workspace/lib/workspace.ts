import type { Highlight } from "./highlight";

export type Thread = {
  thread_id: string;
  title: string;
  external_ref: string | null;
  allowed_version_keys: string[];
  messages?: { role: string; content: string }[];
};

export type Paper = {
  document_version_key: string;
  title: string;
  doi?: string | null;
};

export type WorkspaceSettings = { inference_approved: boolean; run_cost_budget_usd: number };
export type RunStatus = "queued" | "running" | "interrupted" | "completed" | "abstained" | "cancelled" | "failed";

export type RunState = {
  run_id: string;
  question: string;
  status: RunStatus;
  cancel_requested?: boolean;
  created_at: string;
};

export const isActive = (status: RunStatus) =>
  status === "queued" || status === "running" || status === "interrupted";

export type Citation = {
  passage_key: string;
  span_key?: string;
  document_title?: string;
  start_char?: number;
  end_char?: number;
  verbatim_quote: string;
  claim_text?: string;
  numeric_provenance?: string;
  semantic_support?: string;
};

export type Run = RunState & {
  answer_markdown: string | null;
  verified_citations: Citation[];
  rejected_citations?: unknown[];
  proposed_findings: { finding_id: string; missing_information?: string[]; uncertainty_notes?: string[] }[];
  study_comparison: {
    studies: {
      title: string;
      study_design: string;
      population: string;
      intervention: string;
      measurement_protocol: string;
      reported_results: string[];
      missing_information: string[];
      supporting_passage_keys: string[];
    }[];
    evidence_compatibility: string;
    compatibility_rationale: string;
    limitation_notice: string;
  } | null;
  error_message?: string | null;
  abstention_reason?: string | null;
  telemetry?: {
    literature_acquisition?: {
      status: string;
      discovered_records: number;
      unique_works: number;
      ingested: { doi?: string | null; pmcid?: string | null; title: string }[];
      rejected: unknown[];
      provider_failures?: { provider: string }[];
      already_in_corpus?: unknown[];
      origin?: string;
      query_ladder?: { level: number; origin?: string; europe_pmc: string }[];
      projection?: {
        collection: string;
        passage_count: number;
        embedded_now?: number;
        reused_vectors?: number;
      } | null;
    } | null;
    retrieval_candidates?: {
      passage_key: string;
      title: string;
      section_title?: string | null;
      excerpt: string;
      admitted?: boolean;
    }[];
    model_calls?: number;
    model_name?: string;
    estimated_cost_usd?: number;
    autonomous_tool_decisions?: number;
    synthesis_source?: string | null;
  };
};

export type Evidence = {
  passage: {
    title: string;
    text: string;
    doi?: string | null;
    section_title?: string | null;
    source_spans: {
      span_key: string;
      paragraph_source_anchor: string;
      start_char: number;
      end_char: number;
    }[];
  };
  expanded_context: unknown;
  highlight?: Highlight | null;
  highlight_error?: string | null;
};

export type SourceSelection = { key: string; citation?: Citation };

export function titleFromQuestion(question: string) {
  const clean = question.replace(/[\x00-\x1f\x7f]+/g, " ").replace(/\s+/g, " ").trim();
  return clean.length > 88 ? clean.slice(0, 87).trimEnd() + "…" : clean;
}

export function threadTitle(thread: Thread) {
  if (thread.title !== "Scientific Research Thread") return thread.title;
  return titleFromQuestion(thread.messages?.find((message) => message.role === "user")?.content ?? "")
    || "Untitled research";
}

export async function workspaceRequest<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    const timeout = AbortSignal.timeout(15_000);
    response = await fetch(path, {
      cache: "no-store",
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      ...(body === undefined ? {} : {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }),
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("The workspace could not reach the research service. Try again.");
  }
  if (response.status === 401) {
    sessionStorage.setItem("dynamisrag:return", window.location.search);
    window.location.assign("/login");
    throw new Error("Sign in again to continue.");
  }
  if (!response.ok) {
    if (response.status === 409) throw new Error("This conversation already has an active run.");
    if (response.status === 429) throw new Error("Too many requests. Wait a moment and try again.");
    throw new Error("The research service could not complete this request. Try again.");
  }
  return response.json() as Promise<T>;
}

export function research<T>(path: string, body?: unknown, signal?: AbortSignal) {
  return workspaceRequest<T>("/api/research/" + path, body, signal);
}
