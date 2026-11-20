/** Intent, not a retrieval result: only an explicit restricted scope may lock papers. */
export type ResearchMode = "automatic" | "restricted";
export type ResearchScope = { research_mode: ResearchMode; allowed_version_keys: string[] };
const versionKey = /^[a-f0-9]{64}$/;

/** The legacy durable store persists immutable allowlist keys; [] means automatic. */
export function modeFromAllowedKeys(keys: readonly string[]): ResearchMode {
  return keys.length === 0 ? "automatic" : "restricted";
}

export function sessionScope(mode: ResearchMode, keys: readonly string[]): ResearchScope {
  if (mode === "automatic") {
    if (keys.length !== 0) throw new Error("Automatic research cannot carry a paper restriction");
    return { research_mode: "automatic", allowed_version_keys: [] };
  }
  if (mode !== "restricted" || keys.length < 1 || keys.length > 25 ||
      keys.some((key) => !versionKey.test(key)) || new Set(keys).size !== keys.length)
    throw new Error("Restricted research requires 1–25 unique canonical paper version keys");
  return { research_mode: "restricted", allowed_version_keys: [...keys] };
}

/** Validate the untrusted browser-to-Eve session boundary, not model tool output. */
export function parseResearchScope(input: Record<string, unknown>): ResearchScope {
  const keys = input.allowed_version_keys ?? [];
  if (!Array.isArray(keys) || keys.some((key) => typeof key !== "string"))
    throw new Error("Invalid selected papers");
  if (input.research_mode === undefined) {
    // Existing automatic callers send {}. Do not accept an implicit hard lock.
    if (keys.length !== 0) throw new Error("Paper restriction requires an explicit mode");
    return sessionScope("automatic", []);
  }
  if (input.research_mode !== "automatic" && input.research_mode !== "restricted")
    throw new Error("Unrecognized research mode");
  return sessionScope(input.research_mode, keys);
}
