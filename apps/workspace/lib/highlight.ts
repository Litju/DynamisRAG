export type Highlight = {
  span_key: string;
  paragraph_key: string;
  start_char: number;
  end_char: number;
  passage_start_char: number;
  passage_end_char: number;
  quote_sha256: string;
  paragraph_text: string;
};

export type Segment = { text: string; marked: boolean };
export type HighlightResult = {
  segments: Segment[];
  valid: boolean;
  reason?: string;
};

/**
 * Render a validated canonical highlight.
 *
 * The browser must never search for `quote` with indexOf: repeated quotations are
 * common and the first match may be the wrong occurrence. The backend returns
 * passage coordinates derived from the stored span_key; this function re-checks
 * the coordinates and exact text before marking anything.
 */
export function highlightSegments(
  text: string,
  highlight: Highlight | null | undefined,
  quote: string,
): HighlightResult {
  if (!highlight) return { segments: [{ text, marked: false }], valid: false, reason: "missing_span" };
  const start = highlight.passage_start_char;
  const end = highlight.passage_end_char;
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    end <= start ||
    end > text.length
  )
    return { segments: [{ text, marked: false }], valid: false, reason: "invalid_coordinates" };
  if (text.slice(start, end) !== quote)
    return { segments: [{ text, marked: false }], valid: false, reason: "text_mismatch" };
  return {
    segments: [
      { text: text.slice(0, start), marked: false },
      { text: quote, marked: true },
      { text: text.slice(end), marked: false },
    ].filter((segment) => segment.text.length > 0),
    valid: true,
  };
}
