import { defineAgent } from "eve";
import { researchDispatch } from "./lib/dispatch";
export default defineAgent({
  model: researchDispatch,
  defaultTools: false,
  modelContextWindowTokens: 32_768,
});
