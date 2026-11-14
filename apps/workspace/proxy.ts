import { NextResponse, type NextRequest } from "next/server";
import { authFailure, authorize, sameOrigin } from "./lib/auth";

export function proxy(request: NextRequest) {
  const auth = authorize(request);
  if (!auth.ok) return authFailure(auth) ?? new Response(null, { status: 401 });
  if (
    !["GET", "HEAD", "OPTIONS"].includes(request.method) &&
    !sameOrigin(request, { requireOrigin: true })
  )
    return new Response(null, { status: 403 });
  const response = NextResponse.next();
  // The page request is the first point where Basic credentials are exchanged
  // for a short-lived signed session, so subsequent API calls avoid scrypt.
  if (auth.setCookie) response.headers.append("set-cookie", auth.setCookie);
  return response;
}

export const config = { matcher: ["/"] };
