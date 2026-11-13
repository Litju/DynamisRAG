import { defineSandbox } from "eve/sandbox";
import { JustBashSandbox } from "eve/sandbox/just-bash";

// Keep unused sandbox support in-process; never provision a billable cloud VM.
export const environment = JustBashSandbox.environment({ autoInstall: false });
export default defineSandbox(() => environment.open());
