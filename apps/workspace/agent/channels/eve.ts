import { eveChannel } from "eve/channels/eve";
import { verifyJwtHmac, extractBearerToken } from "eve/channels/auth";
import { secret } from "../../lib/auth";
import { ownedSession } from "../../lib/python";

export default eveChannel({
  auth: async (request) => {
    const result = await verifyJwtHmac(
      extractBearerToken(request.headers.get("authorization")),
      {
        algorithm: "HS256",
        issuer: "dynamis-eve",
        audiences: ["dynamisrag-agent-v2"],
        secret: secret("DYNAMISRAG_AGENT_IDENTITY_SECRET"),
      },
    );
    if (!result.ok) return null;
    const user = result.sessionAuth.subject;
    if (!user || !/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(user))
      return null;
    const sessionId = new URL(request.url).pathname.match(
      /\/session\/([^/]+)/,
    )?.[1];
    if (sessionId && !(await ownedSession(user, sessionId))) return null;
    return { ...result.sessionAuth, principalId: user, principalType: "user" };
  },
});
