import { clearedSessionCookie, sameOrigin } from "../../../../lib/auth";

export async function POST(request: Request) {
  if (!sameOrigin(request, { requireOrigin: true }))
    return new Response(null, { status: 403 });
  return new Response(null, {
    status: 204,
    headers: {
      "set-cookie": clearedSessionCookie(request),
      "cache-control": "no-store",
    },
  });
}
