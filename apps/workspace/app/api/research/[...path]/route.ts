import { authenticate, sameOrigin, unauthorized } from "../../../../lib/auth";
import { python } from "../../../../lib/python";

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
    !/^(threads(?:\/[A-Za-z0-9_-]+(?:\/runs)?)?|runs\/[A-Za-z0-9_-]+(?:\/(?:events|citations|cancel|resume))?|papers|evidence\/[a-f0-9]{64})$/.test(
      route,
    )
  )
    return new Response(null, { status: 404 });
  try {
    const headers = new Headers();
    const cursor = request.headers.get("last-event-id");
    if (cursor) headers.set("last-event-id", cursor);
    const result = await python(user, route + new URL(request.url).search, {
      method: request.method,
      headers,
      signal: request.signal,
      ...(request.method === "POST" ? { body: await request.text() } : {}),
    });
    return new Response(result.body, {
      status: result.status,
      headers: {
        "content-type":
          result.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
        "x-accel-buffering": "no",
      },
    });
  } catch {
    return Response.json(
      { error: "Research service unavailable" },
      { status: 503 },
    );
  }
}
export const GET = handle;
export const POST = handle;
export const maxDuration = 300;
