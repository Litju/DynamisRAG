import { setTimeout } from "node:timers/promises";
await Promise.all(
  [8018, 8788, 3018].map(async (port) => {
    for (let attempt = 0; attempt < 60; attempt++) {
      try {
        await fetch(`http://127.0.0.1:${port}/`, {
          signal: AbortSignal.timeout(1000),
        });
        return;
      } catch {
        await setTimeout(500);
      }
    }
    throw new Error(`Local integration service ${port} did not start`);
  }),
);
