import { NextResponse, type NextRequest } from "next/server";
import { authenticate, sameOrigin, unauthorized } from "./lib/auth";
export function proxy(request: NextRequest) {
  if (!authenticate(request.headers)) return unauthorized();
  if (
    !["GET", "HEAD", "OPTIONS"].includes(request.method) &&
    !sameOrigin(request)
  )
    return new Response(null, { status: 403 });
  return NextResponse.next();
}
export const config = { matcher: ["/", "/api/:path*"] };
