import { NextResponse, type NextRequest } from "next/server";
import { authorize, sameOrigin } from "./lib/auth";

export function proxy(request: NextRequest) {
  const loopback = process.env.WORKSPACE_LOOPBACK_ORIGIN;
  if (loopback) {
    const canonical = new URL(loopback);
    const host = request.headers.get("host");
    if (host !== canonical.host) {
      const alias = new URL(loopback);
      alias.hostname = "localhost";
      if (
        host === alias.host &&
        ["GET", "HEAD"].includes(request.method) &&
        ["/", "/login"].includes(request.nextUrl.pathname)
      ) {
        canonical.pathname = request.nextUrl.pathname;
        canonical.search = request.nextUrl.search;
        return NextResponse.redirect(canonical);
      }
      return new Response("Open the workspace at " + canonical.origin, { status: 421 });
    }
  }
  if (request.nextUrl.pathname !== "/") {
    if (loopback && /^\/(?:eve|_eve_internal|\.well-known)(?:\/|$)/.test(request.nextUrl.pathname))
      return new Response(null, { status: 404 });
    return NextResponse.next();
  }
  const auth = authorize(request);
  if (!auth.ok) return NextResponse.redirect(new URL("/login", loopback || request.url));
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

export const config = {
  matcher: ["/", "/login", "/api/:path*", "/eve/:path*", "/_eve_internal/:path*", "/.well-known/:path*"],
};
