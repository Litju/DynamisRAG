import { createHmac, scryptSync, timingSafeEqual } from "node:crypto";

export function secret(name: string): string {
  const value = process.env[name];
  if (!value || value.length < 32)
    throw new Error(`${name} must contain at least 32 characters`);
  return value;
}

// Restricted-preview accounts: {user: "salt:hex(scrypt(password,salt,64))"}.
export function authenticate(headers: Headers): string | null {
  const authorization = headers.get("authorization") || "";
  if (!authorization.startsWith("Basic ") || authorization.length > 2048)
    return null;
  const decoded = Buffer.from(authorization.slice(6), "base64").toString();
  const colon = decoded.indexOf(":");
  const user = decoded.slice(0, colon);
  if (colon < 1 || !/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(user))
    return null;
  const accounts = JSON.parse(process.env.WORKSPACE_ACCOUNTS || "{}") as Record<
    string,
    string
  >;
  const [salt, hash] = (
    Object.hasOwn(accounts, user) ? accounts[user] : "missing:"
  ).split(":");
  const actual = scryptSync(decoded.slice(colon + 1), salt, 64);
  const expected = Buffer.from(hash || "", "hex");
  return expected.length === actual.length && timingSafeEqual(expected, actual)
    ? user
    : null;
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

export function sameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  if (!origin) return true;
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
