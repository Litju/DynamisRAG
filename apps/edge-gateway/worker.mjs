const encoder = new TextEncoder();
const noStore = { "cache-control": "no-store" };
const routes =
  /^\/api\/v2\/(threads(?:\/[A-Za-z0-9_-]+(?:\/runs)?)?|runs\/[A-Za-z0-9_-]+(?:\/(events|citations|cancel|resume))?|papers|evidence\/[a-f0-9]{64})$/;

function decode(segment) {
  return Uint8Array.from(
    atob(segment.replaceAll("-", "+").replaceAll("_", "/")),
    (c) => c.charCodeAt(0),
  );
}

export async function verifyIdentity(header, secret, now = Date.now() / 1000) {
  if (
    !secret ||
    secret.length < 32 ||
    !header?.startsWith("Bearer ") ||
    header.length > 4103
  )
    return false;
  try {
    const parts = header.slice(7).split(".");
    if (parts.length !== 3) return false;
    const metadata = JSON.parse(new TextDecoder().decode(decode(parts[0])));
    if (metadata.alg !== "HS256") return false;
    const key = await crypto.subtle.importKey(
      "raw",
      encoder.encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["verify"],
    );
    if (
      !(await crypto.subtle.verify(
        "HMAC",
        key,
        decode(parts[2]),
        encoder.encode(`${parts[0]}.${parts[1]}`),
      ))
    )
      return false;
    const claims = JSON.parse(new TextDecoder().decode(decode(parts[1])));
    return (
      claims.iss === "dynamis-eve" &&
      claims.aud === "dynamisrag-agent-v2" &&
      typeof claims.sub === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(claims.sub) &&
      typeof claims.iat === "number" &&
      Number.isFinite(claims.iat) &&
      typeof claims.exp === "number" &&
      Number.isFinite(claims.exp) &&
      claims.exp > claims.iat &&
      claims.exp - claims.iat <= 300 &&
      claims.iat <= now + 30 &&
      claims.exp >= now - 30
    );
  } catch {
    return false;
  }
}

async function boundedBody(request) {
  if (!request.body) return undefined;
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 65536) throw new Error("body_limit");
      chunks.push(value);
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  const result = new Uint8Array(size);
  let position = 0;
  for (const chunk of chunks) {
    result.set(chunk, position);
    position += chunk.byteLength;
  }
  return result;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!routes.test(url.pathname) || !["GET", "POST"].includes(request.method))
      return new Response(null, { status: 404, headers: noStore });
    const expires = Date.parse(env.DEMO_EXPIRES_AT || "");
    if (
      !Number.isFinite(expires) ||
      expires <= Date.now() ||
      expires - Date.now() > 86400000
    )
      return new Response("Temporary demo is disabled", {
        status: 503,
        headers: noStore,
      });
    if (
      !(await verifyIdentity(
        request.headers.get("authorization"),
        env.IDENTITY_SECRET,
      ))
    )
      return new Response(null, { status: 401, headers: noStore });
    let upstream;
    try {
      upstream = new URL(env.PYTHON_API_ORIGIN);
      if (
        upstream.protocol !== "https:" ||
        upstream.username ||
        upstream.password ||
        upstream.pathname !== "/" ||
        upstream.search ||
        upstream.hash ||
        upstream.hostname === "trycloudflare.com" ||
        upstream.hostname.endsWith(".trycloudflare.com") ||
        upstream.hostname.endsWith(".invalid")
      )
        throw new Error("invalid_origin");
    } catch {
      return new Response("Protected named tunnel is not configured", {
        status: 503,
        headers: noStore,
      });
    }
    if (!env.ACCESS_CLIENT_ID || !env.ACCESS_CLIENT_SECRET)
      return new Response("Tunnel service authentication is missing", {
        status: 503,
        headers: noStore,
      });
    const headers = new Headers({
      authorization: request.headers.get("authorization"),
      "CF-Access-Client-Id": env.ACCESS_CLIENT_ID,
      "CF-Access-Client-Secret": env.ACCESS_CLIENT_SECRET,
    });
    const cursor = request.headers.get("last-event-id");
    if (cursor) {
      if (!/^\d{1,10}$/.test(cursor))
        return new Response(null, { status: 400, headers: noStore });
      headers.set("last-event-id", cursor);
    }
    let body;
    if (request.method === "POST") {
      if (!request.headers.get("content-type")?.startsWith("application/json"))
        return new Response(null, { status: 415, headers: noStore });
      headers.set("content-type", "application/json");
      try {
        body = await boundedBody(request);
      } catch {
        return new Response(null, { status: 413, headers: noStore });
      }
    }
    try {
      const response = await fetch(
        new URL(url.pathname + url.search, upstream),
        {
          method: request.method,
          headers,
          body,
          signal: request.signal,
          redirect: "manual",
        },
      );
      if (response.status >= 300 && response.status < 400)
        return new Response("Upstream authentication redirected", {
          status: 502,
          headers: noStore,
        });
      return new Response(response.body, {
        status: response.status,
        headers: {
          ...noStore,
          "content-type":
            response.headers.get("content-type") || "application/json",
          "x-accel-buffering": "no",
        },
      });
    } catch {
      return new Response("Local backend is unavailable", {
        status: 503,
        headers: noStore,
      });
    }
  },
};
