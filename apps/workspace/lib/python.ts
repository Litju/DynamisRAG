import { identity } from "./auth";

export async function python(
  user: string,
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const origin = process.env.DYNAMISRAG_API_ORIGIN;
  if (!origin) throw new Error("DYNAMISRAG_API_ORIGIN is required");
  const headers = new Headers(init.headers);
  headers.set("authorization", `Bearer ${identity(user)}`);
  if (init.body) headers.set("content-type", "application/json");
  return fetch(new URL(`/api/v2/${path}`, origin), {
    ...init,
    headers,
    cache: "no-store",
  });
}

export async function json<T>(response: Response): Promise<T> {
  if (!response.ok)
    throw new Error(`Research service returned ${response.status}`);
  return response.json() as Promise<T>;
}

export type Thread = {
  thread_id: string;
  title: string;
  external_ref: string | null;
  allowed_version_keys: string[];
};
export async function ownedSession(
  user: string,
  sessionId: string,
): Promise<Thread | undefined> {
  const threads = await json<Thread[]>(
    await python(user, `threads?external_ref=${encodeURIComponent(sessionId)}`),
  );
  return threads.find((t) => t.external_ref === sessionId);
}
