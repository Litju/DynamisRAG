import { authorize, sameOrigin, withSession } from "../../lib/auth";

function form(error = "", status = 200, retryAfter?: number) {
  const credentialLabel = process.env.WORKSPACE_LOOPBACK_ORIGIN ? "owner credentials" : "preview credentials";
  return new Response(`<!doctype html><html lang="en"><head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sign in · DynamisRAG</title><style>
    body{font:16px system-ui;background:#fafbf8;color:#263e37;margin:0;padding:24px}
    main{max-width:360px;margin:10vh auto}label,input,button{display:block;width:100%;box-sizing:border-box}
    input,button{font:inherit;padding:12px;margin:8px 0 20px;border:1px solid #879b8e;border-radius:6px}
    button{background:#263e37;color:white;cursor:pointer}:focus-visible{outline:3px solid #b07f24}
    </style></head><body><main><h1>Sign in to DynamisRAG</h1>
    <p>Use your ${credentialLabel} to open the research workspace.</p>
    ${error ? `<p role="alert">${error}</p>` : ""}
    <form method="post" action="/login">
    <label for="username">Username</label><input id="username" name="username" autocomplete="username" maxlength="128" required>
    <label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" maxlength="1024" required>
    <button type="submit">Sign in</button></form></main></body></html>`, {
    status,
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
      ...(retryAfter ? { "retry-after": String(retryAfter) } : {}),
    },
  });
}

export function GET() { return form(); }

export async function POST(request: Request) {
  if (!sameOrigin(request, { requireOrigin: true }))
    return new Response(null, { status: 403 });
  if (!request.headers.get("content-type")?.startsWith("application/x-www-form-urlencoded"))
    return new Response(null, { status: 415 });
  const body = await request.text();
  if (body.length > 16384) return new Response(null, { status: 413 });
  const values = new URLSearchParams(body);
  const username = values.get("username") || "";
  const password = values.get("password") || "";
  if (!/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(username) || password.length > 1024)
    return form("Check your username and password.", 400);
  const headers = new Headers(request.headers);
  headers.delete("cookie");
  headers.set("authorization", `Basic ${Buffer.from(`${username}:${password}`).toString("base64")}`);
  const auth = authorize(new Request(request.url, { headers }));
  if (!auth.ok) return auth.status === 429
    ? form(`Too many attempts. Wait ${auth.retryAfter ?? 60} seconds before trying again.`, 429, auth.retryAfter ?? 60)
    : form("Username or password is incorrect.", 401);
  return withSession(new Response(null, {
    status: 303, headers: { location: "/", "cache-control": "no-store" },
  }), auth);
}
