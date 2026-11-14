import { setTimeout as delay } from "node:timers/promises";
import { defineTool } from "eve/tools";
import { z } from "zod";
import { researchEvents } from "../../lib/events";
import { json, ownedSession, python } from "../../lib/python";

type Run = {
  run_id: string;
  status: string;
  answer_markdown: string | null;
  abstention_reason: string | null;
  error_message: string | null;
};
export default defineTool({
  description:
    "Run the authenticated scientific research workflow, including follow-ups and cross-paper comparisons.",
  inputSchema: z.object({ question: z.string().min(1).max(16000) }),
  async *execute({ question }, ctx) {
    const user = ctx.session.auth.current?.principalId;
    if (!user || user !== ctx.session.auth.initiator?.principalId)
      throw new Error("Session owner required");
    const thread = await ownedSession(user, ctx.session.id);
    if (!thread) throw new Error("Research thread binding missing");
    const run = await json<Run>(
      await python(user, `threads/${thread.thread_id}/runs`, {
        method: "POST",
        body: JSON.stringify({
          question,
          idempotency_key: ctx.session.turn.id,
        }),
        signal: ctx.abortSignal,
      }),
    );
    let terminal = false;
    let cursor = 0;
    let streamFailures = 0;
    try {
      while (!terminal) {
        ctx.abortSignal.throwIfAborted();
        try {
          const current = await json<Run>(
            await python(user, `runs/${run.run_id}`, {
              signal: ctx.abortSignal,
            }),
          );
          terminal = ["completed", "abstained", "cancelled", "failed"].includes(
            current.status,
          );
          yield {
            thread_id: thread.thread_id,
            run_id: run.run_id,
            status: current.status,
            answer:
              current.answer_markdown ||
              current.abstention_reason ||
              current.error_message ||
              "Research is running",
            origin: "MODEL_PROPOSED",
            review_state: "PROPOSED",
          };
          if (!terminal) {
            const response = await python(
              user,
              `runs/${run.run_id}/events?after_seq=${cursor}`,
              { signal: ctx.abortSignal },
            );
            for await (const event of researchEvents(response)) {
              if (event.seq <= cursor) continue;
              cursor = event.seq;
              streamFailures = 0;
              yield {
                thread_id: thread.thread_id,
                run_id: run.run_id,
                status: current.status,
                phase: event.stage,
                event_type: event.event_type,
                answer: "Research is running",
                origin: "MODEL_PROPOSED",
                review_state: "PROPOSED",
              };
            }
          }
        } catch (error) {
          if (ctx.abortSignal.aborted || ++streamFailures >= 3) throw error;
          await delay(250 * streamFailures, undefined, {
            signal: ctx.abortSignal,
          });
        }
      }
    } finally {
      if (!terminal)
        await json(
          await python(user, `runs/${run.run_id}/cancel`, { method: "POST" }),
        );
    }
  },
});
