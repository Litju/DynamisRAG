// Test-only transport shim: real Workerd gateway -> real loopback Python, never deployed.
import { gateway } from "../worker.mjs";
export default {
  fetch(request, env) {
    return gateway(request, env, (url, init) =>
      fetch(
        new URL(
          new URL(url).pathname + new URL(url).search,
          "http://127.0.0.1:8018",
        ),
        init,
      ),
    );
  },
};
