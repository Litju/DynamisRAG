import {
  authFailure,
  authorize,
  identity,
  sameOrigin,
  withSession,
} from "../../../../lib/auth";
import { json, ownedSession, python } from "../../../../lib/python";
import { parseResearchScope } from "../../../../lib/research-scope";

async function handle(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const auth = authorize(request);
  if (!auth.ok) return authFailure(auth) ?? new Response(null, { status: 401 });
  if (
    request.method !== "GET" &&
    !sameOrigin(request, { requireOrigin: true })
  )
    return new Response(null, { status: 403 });
  const user = auth.user;
  const { path } = await context.params;
  const route = path.join("/");
  if (
    !/^eve\/v1\/session(?:\/[A-Za-z0-9_-]+(?:\/(?:stream|cancel))?)?$/.test(
      route,
    )
  )
    return new Response(null, { status: 404 });
  const sessionId = path[3];
  try {
    if (sessionId && !(await ownedSession(user, sessionId)))
      return new Response(null, { status: 403 });
    let body = request.method === "POST" ? await request.text() : undefined;
    if (request.method === "POST" && sessionId && path.length === 4) {
      let input;
      try {
        input = JSON.parse(body || "{}");
      } catch {
        return new Response(null, { status: 400 });
      }
      if (
        !input ||
        Object.keys(input).some((key) => key !== "message") ||
        typeof input.message !== "string" ||
        !input.message.trim() ||
        input.message.length > 16000
      )
        return Response.json(
          { error: "Only a text message is accepted" },
          { status: 400 },
        );
      body = JSON.stringify({ message: input.message });
    }
    let scope: string[] = [];
    let title: string | undefined;
    if (!sessionId && request.method === "POST") {
      let input: unknown;
      try {
        input = JSON.parse(body || "{}");
      } catch {
        return new Response(null, { status: 400 });
      }
      if (!input || typeof input !== "object" || Array.isArray(input))
        return new Response(null, { status: 400 });
      if (Object.keys(input).some((key) => !["title", "allowed_version_keys", "research_mode"].includes(key)))
        return Response.json({ error: "Prewarm the session before sending" }, { status: 400 });
      if ("title" in input) {
        if (typeof input.title !== "string" || !input.title.trim() ||
            input.title.length > 100 || /[\x00-\x1f\x7f]/.test(input.title))
          return new Response(null, { status: 400 });
        title = input.title;
      }
      try {
        // The browser must explicitly opt into a hard paper restriction. A
        // missing mode remains compatible only with legacy automatic sessions.
        scope = parseResearchScope(input).allowed_version_keys;
      } catch {
        return Response.json(
          { error: "Choose automatic research or explicitly restrict to selected papers" },
          { status: 400 },
        );
      }
      body = "{}";
    }
    const deploymentOrigin = process.env.VERCEL && process.env.VERCEL_URL
      ? "https://" + process.env.VERCEL_URL
      : undefined;
    const origin =
      process.env.EVE_INTERNAL_ORIGIN ||
      deploymentOrigin ||
      (process.env.NODE_ENV === "production" && !process.env.VERCEL
        ? `http://127.0.0.1:${process.env.EVE_NEXT_PRODUCTION_PORT || "4274"}`
        : new URL(request.url).origin);
    const upstreamHeaders = new Headers({
      authorization: `Bearer ${identity(user)}`,
      "content-type": "application/json",
    });
    // Vercel preview protection also applies to the same-origin Eve service.
    const target = new URL(origin).origin;
    const cookie = request.headers.get("cookie");
    if (cookie && target === new URL(request.url).origin)
      upstreamHeaders.set("cookie", cookie);
    // The platform-issued identity authorizes the server-side hop to this deployment.
    if (target === deploymentOrigin) {
      const token = request.headers.get("x-vercel-oidc-token");
      if (token) upstreamHeaders.set("x-vercel-trusted-oidc-idp-token", token);
    }
    const response = await fetch(
      new URL(`/${route}${new URL(request.url).search}`, origin),
      {
        method: request.method,
        body,
        signal: request.signal,
        cache: "no-store",
        redirect: "manual",
        headers: upstreamHeaders,
      },
    );
    if (response.status >= 300 && response.status < 400)
      return Response.json(
        { error: "Preview authentication is required" },
        { status: 401 },
      );
    if (!sessionId && request.method === "POST" && response.ok) {
      const created = await response.json();
      if (typeof created.sessionId !== "string")
        throw new Error("Missing Eve session id");
      const thread = await json(
        await python(user, "threads", {
          method: "POST",
          body: JSON.stringify({
            external_ref: `eve:${created.sessionId}`,
            ...(title === undefined ? {} : { title }),
            allowed_version_keys: scope,
          }),
        }),
      );
      return withSession(
        Response.json({ ...created, thread }, { status: response.status }),
        auth,
      );
    }
    const headers = new Headers({ "cache-control": "no-store" });
    response.headers.forEach((value, name) => {
      if (
        name.startsWith("x-eve-") ||
        ["content-type", "retry-after"].includes(name)
      )
        headers.set(name, value);
    });
    return withSession(
      new Response(response.body, { status: response.status, headers }),
      auth,
    );
  } catch {
    return Response.json(
      { error: "Conversation service unavailable; retry or reconnect" },
      { status: 503 },
    );
  }
}
export const GET = handle;
export const POST = handle;
export const maxDuration = 300;
