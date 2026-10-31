"""RES-145 immutable provisional evidence graph.

Revision ID: 0005_evidence_graph
Revises: 0004_passage_source_spans
"""

from alembic import op

revision = "0005_evidence_graph"
down_revision = "0004_passage_source_spans"
branch_labels = None
depends_on = None

# Explicit SQL keeps the database trust boundary inspectable, including bypass writes.
_FUNCTIONS = r"""
CREATE FUNCTION evidence_json(j jsonb) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
DECLARE result text;
BEGIN
  CASE jsonb_typeof(j)
    WHEN 'object' THEN
      SELECT '{' || coalesce(string_agg(to_jsonb(key)::text || ':' || evidence_json(value),
             ',' ORDER BY key COLLATE "C"), '') || '}' INTO result FROM jsonb_each(j);
    WHEN 'array' THEN
      SELECT '[' || coalesce(string_agg(evidence_json(value), ',' ORDER BY ord), '') || ']'
      INTO result FROM jsonb_array_elements(j) WITH ORDINALITY AS x(value, ord);
    ELSE result := j::text;
  END CASE;
  RETURN result;
END $$;
CREATE FUNCTION evidence_hash(t text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT AS $$ SELECT encode(sha256(convert_to(t, 'UTF8')), 'hex') $$;
CREATE FUNCTION evidence_key(namespace text, j jsonb) RETURNS text
LANGUAGE sql IMMUTABLE STRICT AS $$
 SELECT evidence_hash(namespace || chr(31) || evidence_json(j)) $$;

CREATE FUNCTION evidence_source(pk text, vk text) RETURNS jsonb
LANGUAGE plpgsql STABLE STRICT AS $$
DECLARE p passage; v document_version; a source_artifact; d document;
 s record; spans jsonb := '[]'; rebuilt text := ''; previous uuid; last_end integer := 0;
 n integer := 0; piece text;
BEGIN
 SELECT * INTO STRICT p FROM passage WHERE passage_key = pk;
 SELECT * INTO STRICT v FROM document_version WHERE id = p.document_version_id;
 SELECT * INTO STRICT a FROM source_artifact WHERE id = v.source_artifact_id;
 SELECT * INTO STRICT d FROM document WHERE id = v.document_id;
 IF v.version_key <> vk OR
    a.artifact_key <> evidence_hash(a.source_system || chr(31) || a.source_external_id || chr(31)
                                  || a.content_sha256) OR
    v.version_key <> evidence_hash(d.canonical_key || chr(31) || a.artifact_key || chr(31)
        || v.parser_revision || chr(31) || v.normalizer_revision || chr(31)
        || v.content_fingerprint) OR
    p.passage_key <> evidence_hash(v.version_key || chr(31) || p.chunker_revision || chr(31)
                                  || p.ordinal::text) OR
    evidence_hash(p.text) <> p.content_sha256 THEN
   RAISE EXCEPTION 'invalid canonical passage identity' USING ERRCODE = 'check_violation';
 END IF;
 FOR s IN SELECT ps.*, q.paragraph_key, q.text AS paragraph_text,
                  q.content_sha256 AS paragraph_sha, q.source_anchor,
                  q.document_version_id AS paragraph_version
          FROM passage_source_span ps JOIN paragraph q ON q.id = ps.paragraph_id
          WHERE ps.passage_id = p.id ORDER BY ps.source_order LOOP
   IF s.document_version_id <> v.id OR s.paragraph_version <> v.id OR
      s.source_order <> n OR s.start_char < 0 OR s.end_char <= s.start_char OR
      s.end_char > length(s.paragraph_text) OR
      s.span_key <> evidence_hash(pk || chr(31) || s.source_order::text) OR
      s.paragraph_key <> evidence_hash(vk || chr(31) || s.source_anchor) OR
      s.paragraph_sha <> evidence_hash(s.paragraph_text) THEN
     RAISE EXCEPTION 'invalid canonical span identity' USING ERRCODE = 'check_violation';
   END IF;
   IF previous = s.paragraph_id THEN
     IF s.start_char < last_end OR
        btrim(substring(s.paragraph_text FROM last_end + 1 FOR s.start_char - last_end),
              E' \n\r\t') <> '' THEN
       RAISE EXCEPTION 'overlapping or gapped spans' USING ERRCODE = 'check_violation';
     END IF;
     piece := substring(s.paragraph_text FROM last_end + 1 FOR s.end_char - last_end);
   ELSE
     piece := substring(s.paragraph_text FROM s.start_char + 1 FOR s.end_char - s.start_char);
     IF n > 0 THEN rebuilt := rebuilt || E'\n\n'; END IF;
   END IF;
   rebuilt := rebuilt || piece;
   spans := spans || jsonb_build_array(jsonb_build_array(s.span_key, s.paragraph_key,
                     s.paragraph_sha, s.start_char, s.end_char, s.source_anchor));
   previous := s.paragraph_id; last_end := s.end_char; n := n + 1;
 END LOOP;
 IF n = 0 OR rebuilt <> p.text THEN
   RAISE EXCEPTION 'source reconstruction failed' USING ERRCODE = 'check_violation';
 END IF;
 RETURN jsonb_build_array(vk, a.artifact_key, a.content_sha256, v.content_fingerprint,
                         pk, p.content_sha256, spans);
END $$;

CREATE FUNCTION evidence_validate() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE ns text; expected jsonb; source jsonb; member jsonb; span passage_source_span;
BEGIN
 IF TG_TABLE_NAME = 'evidence_membership' THEN
   SELECT payload->'members'->NEW.ordinal INTO expected FROM evidence_set WHERE key = NEW.set_key;
   IF NEW.ordinal < 0 OR NEW.payload IS DISTINCT FROM expected OR
      NEW.passage_key IS DISTINCT FROM NEW.payload->>'passage_key' OR NOT EXISTS (
        SELECT 1 FROM passage p JOIN document_version v ON v.id = p.document_version_id
        WHERE p.id = NEW.passage_id AND v.id = NEW.version_id
          AND p.passage_key = NEW.passage_key AND v.version_key = NEW.payload->>'version_key'
      ) THEN RAISE EXCEPTION 'inconsistent evidence membership' USING ERRCODE = 'check_violation';
   END IF;
   source := evidence_source(NEW.passage_key, NEW.payload->>'version_key');
   SELECT jsonb_agg(value->0 ORDER BY ord) INTO expected
     FROM jsonb_array_elements(source->6) WITH ORDINALITY AS x(value, ord);
   IF NEW.payload->>'provenance_sha256' IS DISTINCT FROM evidence_key('source-v1', source)
      OR NEW.payload->'span_keys' IS DISTINCT FROM expected THEN
     RAISE EXCEPTION 'invalid evidence provenance' USING ERRCODE = 'check_violation';
   END IF;
 ELSE
   ns := CASE TG_TABLE_NAME WHEN 'claim' THEN 'claim-v1'
         WHEN 'evidence_set' THEN 'evidence-set-v1' ELSE 'evidence-link-v1' END;
   IF NEW.payload->>'schema_revision' IS DISTINCT FROM ns OR
      NEW.key IS DISTINCT FROM evidence_key(ns, NEW.payload) THEN
     RAISE EXCEPTION 'invalid evidence identity' USING ERRCODE = 'check_violation';
   END IF;
   IF TG_TABLE_NAME IN ('claim', 'evidence_link') AND
      coalesce(NEW.payload->>'review_state', '') NOT IN
      ('PROPOSED','AWAITING_REVIEW','REJECTED') THEN
     RAISE EXCEPTION 'unsigned scientific approval' USING ERRCODE = 'check_violation';
   END IF;
   IF TG_TABLE_NAME = 'claim' THEN
     IF coalesce(NEW.payload->>'category', '') NOT IN
         ('DESCRIPTIVE','METHODOLOGICAL','QUANTITATIVE','ASSOCIATIONAL','CAUSAL') OR
        coalesce(NEW.payload->>'origin', '') NOT IN
         ('SOURCE_EXTRACTION','HUMAN_AUTHORED','MODEL_PROPOSED') OR
        length(btrim(coalesce(NEW.payload->>'proposition', ''))) = 0 OR
        length(btrim(coalesce(NEW.payload->>'origin_reference', ''))) = 0 OR
        length(btrim(coalesce(NEW.payload->>'authoring_process', ''))) = 0 THEN
       RAISE EXCEPTION 'invalid scientific claim' USING ERRCODE = 'check_violation';
     END IF;
   END IF;
   IF TG_TABLE_NAME = 'evidence_set' THEN
     IF coalesce(NEW.payload->>'sufficiency', '') NOT IN
          ('UNASSESSED','INSUFFICIENT','CANDIDATE_SUFFICIENT') OR
        length(btrim(coalesce(NEW.payload->>'question_id', ''))) = 0 OR
        length(btrim(coalesce(NEW.payload->>'question', ''))) = 0 OR
        length(btrim(coalesce(NEW.payload->>'corpus_id', ''))) = 0 OR
        length(btrim(coalesce(NEW.payload->>'selection_context', ''))) = 0 OR
        jsonb_typeof(NEW.payload->'members') IS DISTINCT FROM 'array' OR
        jsonb_array_length(NEW.payload->'members') = 0 OR
        (SELECT count(*) <> count(DISTINCT value->>'passage_key')
         FROM jsonb_array_elements(NEW.payload->'members')) THEN
       RAISE EXCEPTION 'invalid evidence members' USING ERRCODE = 'check_violation';
     END IF;
   ELSIF TG_TABLE_NAME = 'evidence_link' THEN
     SELECT payload INTO member FROM evidence_membership
       WHERE set_key = NEW.set_key AND passage_key = NEW.passage_key;
     SELECT * INTO span FROM passage_source_span WHERE id = NEW.span_id;
     IF member IS NULL OR NEW.payload->'member' IS DISTINCT FROM member OR
        NEW.payload->>'claim_key' IS DISTINCT FROM NEW.claim_key OR
        NEW.payload->>'evidence_set_key' IS DISTINCT FROM NEW.set_key OR
        NEW.payload->>'span_key' IS DISTINCT FROM span.span_key OR
        NOT (member->'span_keys' ? span.span_key) OR NOT EXISTS (
          SELECT 1 FROM evidence_membership m WHERE m.set_key = NEW.set_key
            AND m.passage_key = NEW.passage_key AND m.passage_id = span.passage_id
            AND m.version_id = span.document_version_id
        ) OR coalesce(NEW.payload->>'relation', '') NOT IN
           ('SUPPORTS','CONTRADICTS','CONTEXTUAL','INSUFFICIENT') OR
        length(btrim(coalesce(NEW.payload->>'rationale', ''))) = 0 THEN
       RAISE EXCEPTION 'invalid evidence link' USING ERRCODE = 'check_violation';
     END IF;
   END IF;
 END IF;
 RETURN NEW;
END $$;

CREATE FUNCTION evidence_complete() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE members jsonb;
BEGIN
 SELECT jsonb_agg(payload ORDER BY ordinal) INTO members
 FROM evidence_membership WHERE set_key = NEW.key;
 IF members IS DISTINCT FROM NEW.payload->'members' THEN
   RAISE EXCEPTION 'incomplete evidence set' USING ERRCODE = 'check_violation';
 END IF;
 RETURN NULL;
END $$;
"""


def upgrade() -> None:
    op.execute("""
    CREATE TABLE evidence_set (key text PRIMARY KEY, payload jsonb NOT NULL);
    CREATE TABLE claim (key text PRIMARY KEY, payload jsonb NOT NULL);
    CREATE TABLE evidence_membership (
      set_key text REFERENCES evidence_set(key), passage_key text NOT NULL,
      ordinal integer NOT NULL, passage_id uuid NOT NULL REFERENCES passage(id),
      version_id uuid NOT NULL REFERENCES document_version(id), payload jsonb NOT NULL,
      PRIMARY KEY (set_key, passage_key),
      CONSTRAINT uq_evidence_member_order UNIQUE(set_key, ordinal));
    CREATE TABLE evidence_link (
      key text PRIMARY KEY, claim_key text NOT NULL REFERENCES claim(key),
      set_key text NOT NULL REFERENCES evidence_set(key), passage_key text NOT NULL,
      span_id uuid NOT NULL REFERENCES passage_source_span(id), payload jsonb NOT NULL,
      CONSTRAINT fk_evidence_link_member FOREIGN KEY (set_key, passage_key)
        REFERENCES evidence_membership(set_key, passage_key));
    CREATE INDEX ix_evidence_link_claim ON evidence_link(claim_key);
    """)
    op.execute(_FUNCTIONS)
    for table in ("evidence_set", "claim", "evidence_membership", "evidence_link"):
        op.execute(f"""
        CREATE TRIGGER trg_{table}_validate BEFORE INSERT ON {table}
        FOR EACH ROW EXECUTE FUNCTION evidence_validate();
        CREATE TRIGGER trg_{table}_immutable BEFORE UPDATE OR DELETE ON {table}
        FOR EACH ROW EXECUTE FUNCTION dynamisrag_enforce_immutable();
        """)
    op.execute("""
      CREATE CONSTRAINT TRIGGER trg_evidence_set_complete AFTER INSERT ON evidence_set
      DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION evidence_complete();
    """)


def downgrade() -> None:
    for table in ("evidence_link", "evidence_membership", "claim", "evidence_set"):
        op.execute(f"DROP TABLE {table}")
    for function in (
        "evidence_complete()",
        "evidence_validate()",
        "evidence_source(text,text)",
        "evidence_key(text,jsonb)",
        "evidence_hash(text)",
        "evidence_json(jsonb)",
    ):
        op.execute(f"DROP FUNCTION {function}")
