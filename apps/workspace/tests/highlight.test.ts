import { test } from "node:test";
import assert from "node:assert/strict";
import { highlightSegments, type Highlight } from "../lib/highlight";

const base = (overrides: Partial<Highlight> = {}): Highlight => ({
  span_key: "a".repeat(64),
  paragraph_key: "b".repeat(64),
  start_char: 0,
  end_char: 10,
  passage_start_char: 0,
  passage_end_char: 10,
  quote_sha256: "c".repeat(64),
  paragraph_text: "",
  ...overrides,
});

test("repeated quotations highlight the stored span, never the first indexOf match", () => {
  const text = "jump height increased in trial one. Then jump height increased again.";
  const second = text.lastIndexOf("jump height increased");
  const result = highlightSegments(
    text,
    base({ passage_start_char: second, passage_end_char: second + 21 }),
    "jump height increased",
  );
  assert.equal(result.valid, true);
  const marked = result.segments.filter((segment) => segment.marked);
  assert.equal(marked.length, 1);
  assert.equal(result.segments.map((s) => s.text).join(""), text);
  assert.equal(result.segments[0].text.length, second);
});

test("incorrect offsets are rejected without highlighting another occurrence", () => {
  const text = "jump height increased in trial one. Then jump height increased again.";
  const shifted = text.lastIndexOf("jump height increased") + 1;
  const result = highlightSegments(
    text,
    base({ passage_start_char: shifted, passage_end_char: shifted + 21 }),
    "jump height increased",
  );
  assert.equal(result.valid, false);
  assert.equal(result.reason, "text_mismatch");
  assert.deepEqual(result.segments, [{ text, marked: false }]);
});

test("missing spans and out-of-range coordinates fail closed", () => {
  const text = "short";
  assert.equal(highlightSegments(text, null, "short").valid, false);
  const result = highlightSegments(
    text,
    base({ passage_start_char: 3, passage_end_char: 99 }),
    "short",
  );
  assert.equal(result.valid, false);
  assert.equal(result.reason, "invalid_coordinates");
});

test("a quote that no longer matches the stored text is not highlighted", () => {
  const text = "the intervention improved jump height";
  const result = highlightSegments(
    text,
    base({ passage_start_char: 0, passage_end_char: 11 }),
    "jump height",
  );
  assert.equal(result.valid, false);
  assert.equal(result.reason, "text_mismatch");
});
