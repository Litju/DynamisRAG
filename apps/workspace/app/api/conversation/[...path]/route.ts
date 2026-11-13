import {
  authenticate,
  identity,
  sameOrigin,
  unauthorized,
} from "../../../../lib/auth";
import { json, ownedSession, python } from "../../../../lib/python";

async function handle(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const user = authenticate(request.headers);
  if (!user) return unauthorized();
  if (request.method !== "GET" && !sameOrigin(request))
    return new Response(null, { status: 403 });
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
    let scope: string[] = [];
    if (!sessionId && body) {
      const input = JSON.parse(body);
      // Create first, bind ownership and document scope, then send the first turn.
      if (input.message)
        return Response.json(
          { error: "Prewarm the session before sending" },
          { status: 400 },
        );
      scope = input.allowed_version_keys || [];
      if (
        !Array.isArray(scope) ||
        scope.length > 25 ||
        scope.some((k) => typeof k !== "string" || !/^[a-f0-9]{64}$/.test(k))
      )
        return new Response(null, { status: 400 });
      body = "{}";
    }
    const origin =
      process.env.EVE_INTERNAL_ORIGIN || new URL(request.url).origin;
    const response = await fetch(
      new URL(`/${route}${new URL(request.url).search}`, origin),
      {
        method: request.method,
        body,
        signal: request.signal,
        cache: "no-store",
        headers: {
          authorization: `Bearer ${identity(user)}`,
          "content-type": "application/json",
        },
      },
    );
    if (!sessionId && response.ok) {
      const created = await response.json();
      if (typeof created.sessionId !== "string")
        throw new Error("Missing Eve session id");
      await json(
        await python(user, "threads", {
          method: "POST",
          body: JSON.stringify({
            external_ref: created.sessionId,
            allowed_version_keys: scope,
          }),
        }),
      );
      return Response.json(created, { status: response.status });
    }
    const headers = new Headers({ "cache-control": "no-store" });
    response.headers.forEach((value, name) => {
      if (
        name.startsWith("x-eve-") ||
        ["content-type", "retry-after"].includes(name)
      )
        headers.set(name, value);
    });
    return new Response(response.body, { status: response.status, headers });
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
