import { createServer } from "node:http";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";
import { boundedBody, routes, verifyIdentity } from "./worker.mjs";

const json = { "content-type": "application/json", "cache-control": "no-store" };

const MAX_EXPOSURE_MS = 7 * 86_400_000;

// Public exposure fails closed: the operator must set an expiry no more than seven days out.
export function createFunnelServer({ origin, secret, expiresAt, fetchOrigin = fetch, now = Date.now }) {
  if (!secret || secret.length < 32) throw new Error("Preview identity key is required");
  if (new URL(origin).origin !== origin) throw new Error("Preview backend needs a bare origin");
  const expires = Date.parse(expiresAt || "");
  if (!Number.isFinite(expires) || expires <= now() || expires - now() > MAX_EXPOSURE_MS)
    throw new Error("Preview expiry must be within the next seven days");

  return createServer({ requestTimeout: 15_000, headersTimeout: 10_000 }, async (request, response) => {
    const end = (status) => response.writeHead(status, json).end("{}");
    if (now() >= expires) return end(503);
    if (!request.url?.startsWith("/") || request.url.startsWith("//")) return end(404);
    const url = new URL(request.url, origin);
    if (url.origin !== origin || !routes.test(url.pathname) ||
        !["GET", "POST"].includes(request.method)) return end(404);

    const authorization = request.headers.authorization;
    if (!(await verifyIdentity(authorization, secret))) return end(401);

    const headers = new Headers({ authorization });
    const cursor = request.headers["last-event-id"];
    if (cursor !== undefined) {
      if (typeof cursor !== "string" || !/^\d{1,10}$/.test(cursor)) return end(400);
      headers.set("last-event-id", cursor);
    }
    const controller = new AbortController();
    response.once("close", () => controller.abort());

    let body;
    if (request.method === "POST") {
      if (!request.headers["content-type"]?.startsWith("application/json")) return end(415);
      try {
        body = await boundedBody(new Request(url, {
          method: "POST", body: Readable.toWeb(request), duplex: "half",
        }));
      } catch {
        return end(413);
      }
      headers.set("content-type", "application/json");
    }

    try {
      const upstream = await fetchOrigin(url, {
        method: request.method, headers, body, redirect: "error",
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(270_000)]),
      });
      response.writeHead(upstream.status, {
        "content-type": upstream.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
        "x-accel-buffering": "no",
      });
      response.flushHeaders();
      if (!upstream.body) return response.end();
      Readable.fromWeb(upstream.body).on("error", () => response.destroy()).pipe(response);
    } catch {
      if (!response.destroyed) end(503);
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  createFunnelServer({
    origin: process.env.PYTHON_API_ORIGIN,
    secret: process.env.IDENTITY_SECRET,
    expiresAt: process.env.PREVIEW_EXPIRES_AT,
  }).listen(8080, "0.0.0.0");
}

