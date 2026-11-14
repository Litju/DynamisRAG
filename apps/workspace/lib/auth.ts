import {
  createHash,
  createHmac,
  scryptSync,
  timingSafeEqual,
} from "node:crypto";

export const SESSION_COOKIE = "dynamis_session";
export const SESSION_TTL_SECONDS = 900;
const SESSION_MAX_TTL_SECONDS = 900;
const SESSION_VERSION = "v1";
const MAX_FAILED_ATTEMPTS = 5;
const MAX_FAILED_ATTEMPTS_PER_IP = 20;
const ATTEMPT_WINDOW_MS = 60_000;
const MAX_TRACKED_FAILURES = 10_000;

export function secret(name: string): string {
  const value = process.env[name];
  if (!value || value.length < 32)
    throw new Error(`${name} must contain at least 32 characters`);
  return value;
}

function accounts(): Record<string, string> {
  return JSON.parse(process.env.WORKSPACE_ACCOUNTS || "{}") as Record<
    string,
    string
  >;
}

export function authenticateBasic(input: Headers): string | null {
  const authorization = input.get("authorization") || "";
  if (!authorization.startsWith("Basic ") || authorization.length > 2048)
    return null;
  const decoded = Buffer.from(authorization.slice(6), "base64").toString();
  const colon = decoded.indexOf(":");
  const user = decoded.slice(0, colon);
  if (colon < 1 || !/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(user))
    return null;
  const all = accounts();
  const [salt, hash] = (
    Object.hasOwn(all, user) ? all[user] : "missing:"
  ).split(":");
  const actual = scryptSync(decoded.slice(colon + 1), salt, 64);
  const expected = Buffer.from(hash || "", "hex");
  return expected.length === actual.length && timingSafeEqual(expected, actual)
    ? user
    : null;
}

/** Backward-compatible Basic-only check used by unit tests and trusted callers. */
export function authenticate(input: Headers): string | null {
  return authenticateBasic(input);
}

function sessionKey(): Buffer {
  const source =
    process.env.WORKSPACE_SESSION_SECRET ||
    process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET ||
    "";
  if (source.length < 32)
    throw new Error("WORKSPACE_SESSION_SECRET must contain at least 32 characters");
  // Domain separation: the same shared secret must not sign two token types.
  return createHash("sha256")
    .update(`dynamisrag-workspace-session-v1\0${source}`)
    .digest();
}

function credentialVersion(user: string): string | null {
  const hash = accounts()[user];
  if (!hash) return null;
  return createHash("sha256").update(`${user}\0${hash}`).digest("hex").slice(0, 16);
}

function sign(payload: string): string {
  return createHmac("sha256", sessionKey())
    .update(`${SESSION_VERSION}.${payload}`)
    .digest("base64url");
}

export function createSession(
  user: string,
  nowMs: number = Date.now(),
): string {
  const version = credentialVersion(user);
  if (!version) throw new Error("Unknown account");
  const issuedAt = Math.floor(nowMs / 1000);
  const payload = Buffer.from(
    JSON.stringify({
      sub: user,
      iat: issuedAt,
      exp: issuedAt + SESSION_TTL_SECONDS,
      ver: version,
    }),
  ).toString("base64url");
  return `${SESSION_VERSION}.${payload}.${sign(payload)}`;
}

export function verifySession(
  token: string,
  nowMs: number = Date.now(),
): string | null {
  const parts = token.split(".");
  if (parts.length !== 3 || parts[0] !== SESSION_VERSION) return null;
  const [, payload, signature] = parts;
  const expected = Buffer.from(sign(payload));
  const actual = Buffer.from(signature);
  if (expected.length !== actual.length || !timingSafeEqual(expected, actual))
    return null;
  let claims: unknown;
  try {
    claims = JSON.parse(Buffer.from(payload, "base64url").toString());
  } catch {
    return null;
  }
  if (!claims || typeof claims !== "object") return null;
  const candidate = claims as Record<string, unknown>;
  if (
    typeof candidate.sub !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(candidate.sub) ||
    typeof candidate.iat !== "number" ||
    typeof candidate.exp !== "number" ||
    typeof candidate.ver !== "string"
  )
    return null;
  const now = Math.floor(nowMs / 1000);
  if (
    candidate.iat > now + 30 ||
    candidate.exp <= now ||
    candidate.exp - candidate.iat > SESSION_MAX_TTL_SECONDS ||
    candidate.exp - candidate.iat <= 0
  )
    return null;
  const version = credentialVersion(candidate.sub);
  if (!version || version !== candidate.ver) return null;
  return candidate.sub;
}

function cookieValue(header: string | null, name: string): string | null {
  if (!header) return null;
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return null;
}

function secureCookie(request: Request): boolean {
  return (
    process.env.NODE_ENV === "production" ||
    new URL(request.url).protocol === "https:"
  );
}

export function sessionCookie(
  token: string,
  request: Request,
): string {
  return [
    `${SESSION_COOKIE}=${token}`,
    "HttpOnly",
    "Path=/",
    "SameSite=Lax",
    `Max-Age=${SESSION_TTL_SECONDS}`,
    secureCookie(request) ? "Secure" : "",
  ]
    .filter(Boolean)
    .join("; ");
}

export function clearedSessionCookie(request: Request): string {
  return [
    `${SESSION_COOKIE}=`,
    "HttpOnly",
    "Path=/",
    "SameSite=Lax",
    "Max-Age=0",
    secureCookie(request) ? "Secure" : "",
  ]
    .filter(Boolean)
    .join("; ");
}

type Failure = { key: string; at: number };
const failures: Failure[] = [];

function clientAddress(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0].trim().slice(0, 64);
  return request.headers.get("x-real-ip")?.slice(0, 64) || "unknown";
}

function recentFailures(now: number): Failure[] {
  const cutoff = now - ATTEMPT_WINDOW_MS;
  while (failures.length > 0 && failures[0].at < cutoff) failures.shift();
  return failures;
}

function rateLimited(
  key: string,
  ip: string,
  now: number,
): { limited: boolean; retryAfter: number } {
  const recent = recentFailures(now);
  const forUser = recent.filter((failure) => failure.key === key);
  const forIp = recent.filter((failure) => failure.key === `ip:${ip}`);
  if (
    forUser.length < MAX_FAILED_ATTEMPTS &&
    forIp.length < MAX_FAILED_ATTEMPTS_PER_IP
  )
    return { limited: false, retryAfter: 0 };
  const oldest = Math.min(
    ...forUser.slice(-MAX_FAILED_ATTEMPTS).map((failure) => failure.at),
    ...forIp.slice(-MAX_FAILED_ATTEMPTS_PER_IP).map((failure) => failure.at),
  );
  return {
    limited: true,
    retryAfter: Math.max(1, Math.ceil((oldest + ATTEMPT_WINDOW_MS - now) / 1000)),
  };
}

function recordFailure(key: string, ip: string, now: number): void {
  failures.push({ key, at: now }, { key: `ip:${ip}`, at: now });
  if (failures.length > MAX_TRACKED_FAILURES) {
    failures.splice(0, failures.length - MAX_TRACKED_FAILURES);
  }
}

export function resetAuthAttempts(): void {
  failures.length = 0;
}

export type AuthOutcome =
  | { ok: true; user: string; setCookie?: string }
  | { ok: false; status: 401 | 429; retryAfter?: number };

export function authorize(
  request: Request,
  nowMs: number = Date.now(),
): AuthOutcome {
  const existing = cookieValue(
    request.headers.get("cookie"),
    SESSION_COOKIE,
  );
  if (existing) {
    const user = verifySession(existing, nowMs);
    if (user) return { ok: true, user };
  }
  const authorization = request.headers.get("authorization") || "";
  if (!authorization.startsWith("Basic ")) {
    return { ok: false, status: 401 };
  }
  const decoded = Buffer.from(authorization.slice(6), "base64").toString();
  const user = decoded.slice(0, decoded.indexOf(":"));
  const ip = clientAddress(request);
  const key = `user:${user}`;
  const limit = rateLimited(key, ip, nowMs);
  if (limit.limited)
    return { ok: false, status: 429, retryAfter: limit.retryAfter };
  const authenticated = authenticateBasic(request.headers);
  if (!authenticated) {
    recordFailure(key, ip, nowMs);
    return { ok: false, status: 401 };
  }
  return {
    ok: true,
    user: authenticated,
    setCookie: sessionCookie(createSession(authenticated, nowMs), request),
  };
}

export function identity(user: string): string {
  const now = Math.floor(Date.now() / 1000);
  const header = Buffer.from(
    JSON.stringify({ alg: "HS256", typ: "JWT" }),
  ).toString("base64url");
  const body = Buffer.from(
    JSON.stringify({
      sub: user,
      iss: "dynamis-eve",
      aud: "dynamisrag-agent-v2",
      iat: now,
      exp: now + 120,
    }),
  ).toString("base64url");
  return `${header}.${body}.${createHmac("sha256", secret("DYNAMISRAG_AGENT_IDENTITY_SECRET")).update(`${header}.${body}`).digest("base64url")}`;
}

export function sameOrigin(
  request: Request,
  options: { requireOrigin?: boolean } = {},
): boolean {
  const origin = request.headers.get("origin");
  if (!origin) {
    // Cookie-authenticated unsafe requests must prove same-origin intent. A
    // browser sends Origin on cross-site form/fetch POSTs; Sec-Fetch-Site covers
    // same-origin requests where a proxy stripped Origin.
    return options.requireOrigin === true
      ? request.headers.get("sec-fetch-site") === "same-origin"
      : true;
  }
  try {
    const parsed = new URL(origin);
    return (
      ["http:", "https:"].includes(parsed.protocol) &&
      parsed.host === (request.headers.get("host") || new URL(request.url).host)
    );
  } catch {
    return false;
  }
}

export const unauthorized = () =>
  Response.json(
    { error: "Sign in to the research workspace" },
    {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="DynamisRAG", charset="UTF-8"',
        "Cache-Control": "no-store",
      },
    },
  );

export const rateLimitedResponse = (retryAfter: number) =>
  Response.json(
    { error: "Too many authentication attempts; retry later" },
    {
      status: 429,
      headers: {
        "Retry-After": String(retryAfter),
        "Cache-Control": "no-store",
      },
    },
  );

export function authFailure(outcome: AuthOutcome): Response | null {
  if (outcome.ok) return null;
  return outcome.status === 429
    ? rateLimitedResponse(outcome.retryAfter ?? 60)
    : unauthorized();
}

export function withSession(
  response: Response,
  outcome: AuthOutcome,
): Response {
  if (outcome.ok && outcome.setCookie) {
    response.headers.append("set-cookie", outcome.setCookie);
  }
  return response;
}
