"""The frozen RES-138 benchmark contracts: workloads, candidates and identity.

This module owns everything the benchmark is *not allowed to decide later*. It
is a declaration of what a result would have to have been produced from, so that
a reader of a finished artifact can tell whether the evidence supports a claim
without trusting the run that produced it.

Four separable concerns, kept apart for the same reason the embedding boundary
keeps them apart — a fingerprint that merges the wrong two lies:

* **workload value types** (:class:`RetrievalDocument`, :class:`RetrievalQuery`,
  :class:`RetrievalQrel`, :class:`RetrievalWorkload`) — what was embedded and
  judged, in one canonical order, with every content digest bound to the exact
  UTF-8 text that was embedded;
* **frozen sources** (:class:`BeirSourceSpec`) — which archives, from where, and
  the SHA-256 each one must hash to;
* **frozen candidates** (:class:`ModelCandidateSpec`) — which weights at which
  immutable revision, with which pooling, truncation boundary and model-native
  prompts;
* **frozen numerics** — dimensions, shard size, retrieval cut-offs, corpus
  chunking, bootstrap parameters, calibration gates and artifact revisions.

**Why these are constants and not parameters.** Every one of them is a choice
that could be made *after* seeing results and would then be unfalsifiable. The
dimension decides what Recall@100 can even distinguish, the prompt decides what
the model retrieves, the shard size decides whether a run can resume, the
bootstrap seed and sample count decide the width of the confidence interval.
All of them are therefore named once, here, and every artifact binds them.

**Canonical order is a contract, not a convention.** Documents ascend by
``document_id``, queries ascend by ``query_id``, qrels ascend by
``(query_id, document_id)``. The order is fixed here rather than produced by a
sort in the middle of a run, because it decides the *request sequence* the model
saw and therefore the shard boundaries: a corpus ordered differently produces
different shards, and a shard identity that did not notice would claim vectors
for rows it does not hold.

**What is deliberately absent.** No default model, no default dimension, no
selected candidate, and nothing that would be read by production code. RES-138
has produced no result yet; the only thing this package may do is make a future
result checkable. ``dynamisrag.embedding`` and ``dynamisrag.search`` do not
import this module, and nothing here encodes a preference between the two
candidates.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final, Self

from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.embedding.contracts import TruncationDirection, canonical_json
from dynamisrag.embedding.errors import EmbeddingContractError
from dynamisrag.embedding.identity import require_embedding_identifier, require_sha256_hex

__all__ = [
    "BEIR_QREL_SPLIT",
    "RES138_ARTIFACT_REVISIONS",
    "RES138_ATTENTION_BACKEND",
    "RES138_BASE_DIMENSION",
    "RES138_BEIR_SOURCES",
    "RES138_BOOTSTRAP_CONFIDENCE",
    "RES138_BOOTSTRAP_SAMPLES",
    "RES138_BOOTSTRAP_SEED",
    "RES138_CALIBRATION_BANDS",
    "RES138_CALIBRATION_ITEMS_PER_CELL",
    "RES138_CALIBRATION_SELECTION_REVISION",
    "RES138_CALIBRATION_TOP_K",
    "RES138_CANDIDATE_DIMENSIONS",
    "RES138_CORPUS_CHUNK_SIZE",
    "RES138_DOCUMENT_TEXT_POLICY",
    "RES138_DRIVE_LOCATIONS",
    "RES138_DRIVE_ROOT",
    "RES138_INPUT_MAX_TOKENS",
    "RES138_INPUT_TRUNCATION_DIRECTION",
    "RES138_LOCAL_SCRATCH_ROOT",
    "RES138_LONG_CONTEXT_STAGE",
    "RES138_MODEL_CANDIDATES",
    "RES138_MODEL_IDS",
    "RES138_MRL_CALIBRATION_GATE",
    "RES138_MRL_DERIVATION_REVISION",
    "RES138_NDCG_CUTOFF",
    "RES138_POOLING_MODES",
    "RES138_PRODUCTION_STAGE",
    "RES138_PROMPT_NAMES",
    "RES138_QUERY_SELECTION_POLICY",
    "RES138_RECALL_CUTOFFS",
    "RES138_REFERENCE_STAGE",
    "RES138_RETRIEVAL_TOP_K",
    "RES138_RUN_ID_PREFIX",
    "RES138_SHARD_SIZE",
    "RES138_SUPPORTED_DTYPES",
    "RES138_WORKLOAD_NAMES",
    "BeirSourceSpec",
    "DriveLocation",
    "ModelCandidateSpec",
    "MrlCalibrationGate",
    "RetrievalDocument",
    "RetrievalPromptSpec",
    "RetrievalQrel",
    "RetrievalQuery",
    "RetrievalWorkload",
    "beir_document_embedding_text",
    "drive_path",
    "ordered_ids_sha256",
    "require_candidate_dimension",
    "require_code_sha",
    "require_exact_bool",
    "require_exact_int",
    "require_exact_str",
    "require_frozen_dtype",
    "require_shard_size",
    "text_sha256",
]

# ---------------------------------------------------------------------------
# Primitive validation
#
# Defined once, here, because every value in this package that reaches a hashed
# payload, an artifact or a model is a primitive, and Python will not stop a
# wrong one from arriving. A type annotation is a promise to a reader and a
# checker; it is not a runtime gate, and these dataclasses are exported, so they
# are constructed directly.
# ---------------------------------------------------------------------------


def require_exact_int(
    value: object, *, kind: str, operation: str, minimum: int, because: str
) -> int:
    """Require a real ``int`` at or above ``minimum``.

    ``bool`` first, because ``True`` is an ``int`` of value 1 and would pass a
    range check as a count of one; then the type, because ``1.5`` passes it too
    and would reach an artifact as ``1.5`` where an integer is meant.

    Takes ``object`` on purpose. The fields it guards are annotated, so a type
    checker already knows the answer and would call the runtime half redundant —
    which is the point: the annotation is a promise to a reader of the source,
    while this is the gate a caller who did not read it still meets.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkContractError(
            f"benchmark {kind} must be an explicit integer, got {value!r} of type "
            f"{type(value).__name__}. {because}",
            operation=operation,
        )
    if value < minimum:
        raise BenchmarkContractError(
            f"benchmark {kind} must be at least {minimum}, got {value}. {because}",
            operation=operation,
        )
    return value


def require_exact_str(value: object, *, kind: str, operation: str) -> str:
    """Require a real ``str``, empty or not, and reject the look-alikes.

    A prompt content field is legitimately the empty string — Qwen's document
    prompt is — so emptiness is not the question here; the type is, because the
    content is hashed into model provenance and a non-string would hash as
    something a reader could not interpret as a prompt.
    """
    if isinstance(value, str):
        return value
    raise BenchmarkContractError(
        f"benchmark {kind} must be a string, got {value!r} of type {type(value).__name__}. The "
        "value is hashed into artifact identity, so its type is part of that identity.",
        operation=operation,
    )


def require_exact_bool(value: object, *, kind: str, operation: str, because: str) -> bool:
    """Require a real ``bool``.

    ``True`` and ``1`` are equal and hash alike in most serialisers, so a flag that
    arrived as the integer ``1`` would be indistinguishable from the boolean in a
    hashed payload while meaning something quite different at the call site: one is a
    policy, the other is a number that happened to be truthy. This gate exists so the
    frozen contract records which of the two was declared.

    ``because`` names what the flag decides, because a bare "must be a bool" does not
    tell a reader which of their two declarations was wrong.
    """
    if isinstance(value, bool):
        return value
    raise BenchmarkContractError(
        f"benchmark {kind} must be true or false, got {value!r} of type {type(value).__name__}. "
        f"{because}",
        operation=operation,
    )


def require_frozen_dtype(value: object, *, kind: str, operation: str, because: str) -> str:
    """Require a dtype name from the closed supported set.

    **A closed set, not a pattern.** Any string shaped like a dtype would pass a
    pattern, and the whole point of freezing the compute dtype is that a run cannot
    quietly execute in a different precision than the plan declares. Widening this set is
    an amendment to the benchmark contract and has to be a visible edit to the constant
    above, not a value that arrives from a config or a CLI flag.
    """
    name = require_exact_str(value, kind=kind, operation=operation)
    if name in RES138_SUPPORTED_DTYPES:
        return name
    raise BenchmarkContractError(
        f"benchmark {kind} is {name!r}, which is not one of the frozen "
        f"{list(RES138_SUPPORTED_DTYPES)}. {because}",
        operation=operation,
    )


# ---------------------------------------------------------------------------
# The three staged concerns of RES-138
#
# RES-138 is a benchmark architecture amendment, not one execution. Reference
# quality, production qualification and long context are separate stages with
# separate evidence, and each stage's identity carries which stage it is so an
# artifact from one can never be read as an artifact from another.
# ---------------------------------------------------------------------------

RES138_REFERENCE_STAGE: Final[str] = "reference-quality"
"""Stage A: the float32 reference-quality benchmark at the frozen 8192 boundary.

Both candidates are encoded by the same native sentence-transformers runtime,
with the same boundary, scheduler and exact retrieval; the evidence this stage
produces is quality evidence and reference execution observations. It is not a
deployment qualification: throughput measured here is never reported as
production throughput, and index footprint is never measured here.
"""

RES138_PRODUCTION_STAGE: Final[str] = "production-qualification"
"""Stage B: production inference qualification against the Stage A reference.

Consumes a Stage A result, runs the candidate under the actual production
inference configuration (TEI with the candidate-supported precision and
backend), and requires an explicit numerical and ranking equivalence gate
against the Stage A reference before any operational metric may be used. The
A100-80GB deployment floor belongs to this stage, not to Stage A.
"""

RES138_LONG_CONTEXT_STAGE: Final[str] = "long-context"
"""Stage C: an optional long-context benchmark (LongEmbed/LoCo-style, 8k/16k/32k).

Defined separately so it cannot block Stage A or Stage B unless an operator
explicitly promotes it to a production requirement. See
:mod:`dynamisrag.benchmark.long_context`.
"""

_STAGES: Final[tuple[str, ...]] = (
    RES138_REFERENCE_STAGE,
    RES138_PRODUCTION_STAGE,
    RES138_LONG_CONTEXT_STAGE,
)
"""Every declared stage, so a payload cannot name a stage nobody defined."""


def require_stage(value: object, *, operation: str) -> str:
    """Require one of the three declared stage names."""
    if isinstance(value, str) and value in _STAGES:
        return value
    raise BenchmarkContractError(
        f"benchmark stage {value!r} is not one of {list(_STAGES)}. A staged result has to name "
        "the stage it belongs to, because evidence from one stage does not answer another "
        "stage's question.",
        operation=operation,
    )


# ---------------------------------------------------------------------------
# Frozen provenance of the three workloads
# ---------------------------------------------------------------------------

RES138_DOCUMENT_TEXT_POLICY: Final[str] = "beir-document-text-v1"
"""How a BEIR ``title``/``text`` pair becomes the one string that is embedded.

Stated as a revision because it changes the vectors. :func:`beir_document_embedding_text`
is the only implementation, and a workload loader must call it rather than
interpolating ``title`` and ``text`` itself — a notebook that invents a
different join would produce a perfectly valid-looking result for a policy the
plan never declared.
"""

RES138_QUERY_SELECTION_POLICY: Final[str] = "beir-judged-queries-v1"
"""Which BEIR queries belong to a workload.

Every query id that appears in the frozen qrels split, and no others. NFCorpus
ships 3,237 query records for 323 judged ones; embedding the 2,914 unjudged
queries would spend GPU time on vectors no metric can use, and *dropping* them
without recording the fact would make the workload unaccountable for. The count
excluded is reported in the source manifest.
"""

BEIR_QREL_SPLIT: Final[str] = "test"
"""The BEIR qrels file every workload is frozen on: ``<name>/qrels/test.tsv``.

``train.tsv`` and ``dev.tsv`` exist for SciFact and NFCorpus and are ignored.
TREC-COVID ships only ``test.tsv``. Choosing the split once, here, is what keeps
three different files from being compared as one benchmark.
"""


@dataclass(frozen=True)
class BeirSourceSpec:
    """One frozen workload archive: where it comes from and what it must hash to.

    ``sha256`` is the only thing that makes a downloaded file the frozen source.
    The filename, the byte size and the folder a copy happens to sit in are all
    forgeable, so acquisition verifies the digest *before* extraction and
    verifies it again on every later run that reads the copy.
    """

    workload: str
    archive_name: str
    url: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if not self.workload:
            raise BenchmarkContractError(
                "a frozen workload source must name the workload it provides; an unnamed source "
                "cannot be attached to a result.",
                operation="beir_source_spec",
            )
        if not self.url.startswith("https://"):
            raise BenchmarkContractError(
                f"frozen workload source {self.workload!r} has URL {self.url!r}, which is not an "
                "https URL. The digest is what verifies the bytes, but a plaintext transport "
                "would let a party in the middle choose which bytes to serve against a digest "
                "this project did not compute.",
                operation="beir_source_spec",
                workload=self.workload,
            )
        try:
            require_sha256_hex(
                self.sha256,
                kind="frozen workload source digest",
                operation="beir_source_spec",
            )
        except EmbeddingContractError:
            raise BenchmarkContractError(
                f"frozen workload source {self.workload!r} declares sha256 {self.sha256!r}, "
                "which is not 64 lowercase hexadecimal characters. A digest, not a name, is what "
                "makes an archive the frozen source.",
                operation="beir_source_spec",
                workload=self.workload,
            ) from None
        byte_size = require_exact_int(
            self.byte_size,
            kind="frozen source byte_size",
            operation="beir_source_spec",
            minimum=1,
            because="A zero-length or negative archive cannot contain a workload, and the size is "
            "recorded for the source manifest rather than substituted for the digest.",
        )
        object.__setattr__(self, "byte_size", byte_size)

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this frozen source."""
        return {
            "workload": self.workload,
            "archive_name": self.archive_name,
            "url": self.url,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


RES138_BEIR_SOURCES: Final[tuple[BeirSourceSpec, ...]] = (
    BeirSourceSpec(
        workload="scifact",
        archive_name="scifact.zip",
        url=("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"),
        sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        byte_size=2_816_079,
    ),
    BeirSourceSpec(
        workload="nfcorpus",
        archive_name="nfcorpus.zip",
        url=("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"),
        sha256="efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        byte_size=2_448_432,
    ),
    BeirSourceSpec(
        workload="trec-covid",
        archive_name="trec-covid.zip",
        url=("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"),
        sha256="120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0",
        byte_size=73_876_720,
    ),
)
"""The three frozen workloads, in a fixed order.

Scientific and evidence retrieval only, as RES-138 was scoped: SciFact (citation
verdict claims), NFCorpus (nutrition-science queries over a graded hierarchy)
and TREC-COVID (a pandemic-information corpus). All three ship as the original
BEIR ZIP distribution, parsed with the standard library — never through a
Hugging Face ``datasets`` parquet conversion, which would substitute a
repackaging of the corpus for the corpus.

The digests were computed from the archives themselves and are the only
authority on their content.
"""

RES138_WORKLOAD_NAMES: Final[tuple[str, ...]] = tuple(
    source.workload for source in RES138_BEIR_SOURCES
)
"""The frozen workload names, in the same order as the sources."""


# ---------------------------------------------------------------------------
# Frozen candidates
# ---------------------------------------------------------------------------

RES138_CANDIDATE_DIMENSIONS: Final[tuple[int, ...]] = (512, 1024)
"""The four evaluated configurations: two models at two dimensions each.

Declared as a set of *candidate* dimensions and nothing more. Neither value is
a default, and the pair is not a preference; RES-139 consumes a selection this
benchmark produces, and until then no dimension in this tuple is chosen.
"""

RES138_BASE_DIMENSION: Final[int] = 1024
"""The dimension generated once and the one every other candidate derives from.

One 1024-dimensional pass per input per model, with 512 obtained by the frozen
Matryoshka prefix rule, rather than four independent passes. That optimisation is
only legitimate if native-512 and derived-512 agree numerically, which
:mod:`dynamisrag.benchmark.mrl` proves on a calibration set before any corpus is
embedded, and which the preflight records as a per-model, per-path decision.
"""

RES138_MRL_DERIVATION_REVISION: Final[str] = "mrl-prefix-renorm-v1"
"""Revision of the derivation: take the first 512 components, renormalise, float32.

Named, versioned and hashed because it is arithmetic applied to vectors: a
different rule (scaling by ``1/sqrt(2)``, normalising the whole vector after
truncation, keeping float64) produces different vectors, so "the 512-d vectors"
is not an identity until the rule is named.
"""

RES138_INPUT_MAX_TOKENS: Final[int] = 8192
"""The one common reference boundary every Stage A candidate, document and query shares.

Frozen for the Stage A reference-quality benchmark at 8192 tokens: low enough
that both candidates execute the frozen schedule in float32 on an ordinary
hosted GPU, and high enough to carry SciFact, NFCorpus and the overwhelming
majority of TREC-COVID. Long context is not part of Stage A; the 8k/16k/32k
windows are a separate optional benchmark (:mod:`dynamisrag.benchmark.long_context`)
and cannot be promoted into this boundary without a plan revision.

Every candidate must declare a native max sequence length **at least** this
value, and the loaded model's own boundary is set to **exactly** it before any
encode: a model whose loaded boundary were longer would silently accept inputs
this contract declares truncated, and one whose boundary were shorter would
truncate at a point nobody declared.

The boundary is not a capacity limit and not a refusal rule: inputs longer than
it are truncated to it, explicitly and on the right, and the raw token count is
still measured and persisted so the truncation is auditable. The scheduler's
token-square budget is this value squared.
"""

RES138_INPUT_TRUNCATION_DIRECTION: Final[str] = TruncationDirection.RIGHT.value
"""Which end of an over-long input the native encoding path keeps.

``right`` means the beginning of the prompt-plus-text is kept and the tail is
discarded. It is the value the frozen inference path actually applies: the
sentence-transformers 5.0.0 ``Transformer.tokenize`` passes
``truncation="longest_first"`` with ``max_length=model.max_seq_length`` to the
tokenizer, and a tokenizer whose ``truncation_side`` is ``right`` keeps the head
of a single sequence. The runner verifies ``tokenizer.truncation_side`` at load,
so this constant and the loaded object are the same policy rather than two
claims.
"""

RES138_ATTENTION_BACKEND: Final[str] = "sdpa"
"""The attention backend both frozen candidates are explicitly loaded and executed in.

**Requested first, then observed.** Every load passes
``model_kwargs={"attn_implementation": RES138_ATTENTION_BACKEND, ...}`` to
``SentenceTransformer`` — the whole policy lives in
:func:`~dynamisrag.benchmark.runner.load_keyword_arguments`, with no
candidate-specific branch — and the loaded Hugging Face model answers for
itself: the runner reads ``auto_model.config._attn_implementation`` (the field
Transformers 4.54.0 settles after ``from_pretrained``) and refuses any value
other than this one before a single calibration, probe or corpus input is
encoded. Both the requested and the observed half are recorded per candidate.

PyTorch is Colab-owned and its implicit dispatch is not an observation, so
recording only a declared value would make "SDPA ran" a claim about a default
rather than a fact about the loaded model. The memory probe's declared
execution conditions and the plan's execution policy carry this value, and a
preflight whose runtime records do not carry both halves cannot authorise a
full run.
"""

_CODE_SHA: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


def require_code_sha(value: str, *, operation: str) -> str:
    """Require exactly 40 lowercase hexadecimal characters — an exact commit.

    GitHub is the only code transport for a Colab run, so the notebook clones
    the repository and checks out one commit. That commit has to be named in
    full: a branch name, a tag, ``main``, ``latest`` or an abbreviated SHA all
    resolve to *some* commit at an unknown time, which is the entire thing the
    run identity exists to prevent.

    40 characters, not 7, and not 64: this is a Git commit id, not a digest, and
    confusing the two would produce a value that looks verifiable and is not.
    """
    if _CODE_SHA.fullmatch(value):
        return value
    raise BenchmarkContractError(
        f"CODE_SHA must be exactly 40 lowercase hexadecimal characters, got {value!r}. A branch "
        "name, a tag, 'main', 'latest' or an abbreviated SHA all resolve to a commit that is not "
        "known until the clone runs, so none of them can bind a benchmark artifact to the code "
        "that produced it. Paste the full SHA reported for the RES-138 harness branch.",
        operation=operation,
        observed=value[:64],
    )


@dataclass(frozen=True)
class RetrievalPromptSpec:
    """One model-native prompt, identified by name and by content.

    The prompt content is part of the artifact identity, so it is hashed as well
    as named. ``name`` is the key in the model's own
    ``config_sentence_transformers.json`` — ``query`` or ``document`` — and it is
    what TEI would receive as ``prompt_name``. ``content`` is the string that key
    resolves to *at the pinned revision*, which the runner re-reads from the
    repository and compares, failing rather than substituting a prompt it invented.
    """

    name: str
    content: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.name not in RES138_PROMPT_NAMES:
            raise BenchmarkContractError(
                f"retrieval prompt name {self.name!r} is not one of the two frozen names "
                f"{list(RES138_PROMPT_NAMES)}; a model-native prompt is addressed by the name the "
                "pinned repository declares, not by a nickname.",
                operation="retrieval_prompt_spec",
                expected=str(sorted(RES138_PROMPT_NAMES)),
                observed=self.name,
            )
        # An empty prompt is legitimate and load-bearing: Qwen's document prompt
        # is the empty string, and it means "no instruction, because the document
        # is not a question". Refusing it would make one of the two candidates
        # unrunnable, so only the type is checked.
        object.__setattr__(
            self,
            "content",
            require_exact_str(
                self.content, kind="retrieval prompt content", operation="retrieval_prompt_spec"
            ),
        )
        try:
            require_sha256_hex(
                self.content_sha256,
                kind="retrieval prompt content digest",
                operation="retrieval_prompt_spec",
            )
        except EmbeddingContractError:
            raise BenchmarkContractError(
                f"retrieval prompt {self.name!r} declares content_sha256 {self.content_sha256!r}, "
                "which is not 64 lowercase hexadecimal characters.",
                operation="retrieval_prompt_spec",
            ) from None

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this prompt identity."""
        return {
            "name": self.name,
            "content": self.content,
            "content_sha256": self.content_sha256,
        }


RES138_PROMPT_NAMES: Final[tuple[str, ...]] = ("query", "document")
"""The only two prompt identities a retrieval benchmark may use.

One per direction. A single generic prompt applied to both sides would be a
policy invented after the fact, which is exactly what the model-native prompts
exist to prevent.
"""


def text_sha256(text: str) -> str:
    """SHA-256 over the exact UTF-8 bytes of ``text``.

    The same definition RES-137 applies to a passage, reused rather than
    reimplemented: a content digest that meant one thing in production and
    another in the benchmark would make the two incomparable, and
    ``dynamisrag.embedding.passage_content_sha256`` is already the definition
    every artifact in this repository is checked against.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelCandidateSpec:
    """One frozen candidate: weights, revision, loading, pooling, boundary and prompts.

    ``revision`` is a 40-character commit, frozen before any result exists. It is
    never resolved at run time: the benchmark does not ask the Hub what the
    branch points at today, because a model identity that follows a branch is not
    an identity.

    **Loading is part of the identity, not a detail of the runner.**
    ``trust_remote_code`` says whether the repository ships its own modelling code
    that ``transformers`` must be allowed to execute, and it is not a preference:
    Voyage 4 Nano cannot be constructed at all without it, while granting it for a
    repository that does not need it would execute unreviewed code from the Hub for
    no reason. It is a property of the frozen candidate, so it is declared here, put
    in the hashed candidate payload, put in the plan, and passed to
    ``SentenceTransformer`` verbatim. A runner that decided it from the model id
    would put a policy about *this* model inside the code that loads *every* model,
    where the next candidate added would silently inherit it.

    ``compute_dtype`` is the dtype the weights are loaded and executed in, and
    ``output_dtype`` is the dtype of the persisted matrix. **They are different
    facts and conflating them produces a provenance lie**: a NumPy artifact cast to
    float32 says nothing about the precision the forward pass ran in. The runner
    requests ``compute_dtype``, then reads the dtype back off the loaded parameters
    and records what it actually observed, so a model that ignored the request is a
    failed preflight rather than a quietly mislabelled artifact.

    ``native_max_sequence_length`` is the model's own truncation boundary, read
    from the pinned repository: ``sentence_bert_config.json`` for Voyage,
    ``config.json``'s ``max_position_embeddings`` for Qwen (whose
    ``tokenizer_config.json`` declares 131072, four times the positions the model
    actually has). It must be at least :data:`RES138_INPUT_MAX_TOKENS`, the one
    common boundary every input is measured and truncated at, and the runner
    requires the *loaded* model to report exactly that common boundary.

    ``sequence_length_source`` is recorded so a reviewer can see where the number
    came from instead of trusting it.
    """

    model_id: str
    revision: str
    license: str
    trust_remote_code: bool
    compute_dtype: str
    output_dtype: str
    pooling_mode: str
    native_max_sequence_length: int
    sequence_length_source: str
    query_prompt: RetrievalPromptSpec
    document_prompt: RetrievalPromptSpec

    def __post_init__(self) -> None:
        try:
            require_embedding_identifier(
                self.model_id, kind="candidate model id", operation="model_candidate_spec"
            )
            require_embedding_identifier(
                self.revision, kind="candidate model revision", operation="model_candidate_spec"
            )
        except EmbeddingContractError as error:
            raise BenchmarkContractError(str(error), operation="model_candidate_spec") from None
        if not _CODE_SHA.fullmatch(self.revision):
            raise BenchmarkContractError(
                f"candidate model {self.model_id!r} declares revision {self.revision!r}, which is "
                "not exactly 40 lowercase hexadecimal characters. A frozen revision is an "
                "immutable commit id; a branch or a tag would let the weights change under the "
                "artifact that names them.",
                operation="model_candidate_spec",
                model_id=self.model_id,
            )
        object.__setattr__(
            self,
            "trust_remote_code",
            require_exact_bool(
                self.trust_remote_code,
                kind="candidate trust_remote_code",
                operation="model_candidate_spec",
                because="It decides whether code from the model repository is executed while "
                "loading the weights, so it is a policy declared for this candidate rather than "
                "a truthy value.",
            ),
        )
        object.__setattr__(
            self,
            "compute_dtype",
            require_frozen_dtype(
                self.compute_dtype,
                kind="candidate compute_dtype",
                operation="model_candidate_spec",
                because="The precision the weights execute in is identity-bearing: the same "
                "weights in two precisions return different vectors, and RES-138 needs one "
                "explicit compute dtype for all four candidate runs to be comparable.",
            ),
        )
        object.__setattr__(
            self,
            "output_dtype",
            require_frozen_dtype(
                self.output_dtype,
                kind="candidate output_dtype",
                operation="model_candidate_spec",
                because="It is the dtype of the persisted matrix, and it is recorded separately "
                "from the compute dtype so the two cannot be mistaken for one another.",
            ),
        )
        if self.pooling_mode not in RES138_POOLING_MODES:
            raise BenchmarkContractError(
                f"candidate model {self.model_id!r} declares pooling mode {self.pooling_mode!r}, "
                f"which is not one of {list(RES138_POOLING_MODES)}. CLS, mean and last-token "
                "pooling over identical weights are vectors in different spaces.",
                operation="model_candidate_spec",
                model_id=self.model_id,
            )
        limit = require_exact_int(
            self.native_max_sequence_length,
            kind="candidate native_max_sequence_length",
            operation="model_candidate_spec",
            minimum=RES138_INPUT_MAX_TOKENS,
            because="Every candidate shares one common input boundary, and a model whose native "
            "boundary is shorter than it cannot legally receive the frozen maximum input.",
        )
        object.__setattr__(self, "native_max_sequence_length", limit)
        if not self.sequence_length_source:
            raise BenchmarkContractError(
                f"candidate model {self.model_id!r} must state where its native max sequence "
                "length was read at the pinned revision, so that the number can be re-checked "
                "rather than trusted.",
                operation="model_candidate_spec",
                model_id=self.model_id,
            )

    @property
    def prompt_sha256(self) -> str:
        """One digest over both prompt identities.

        A model whose query prompt changes and whose document prompt does not is
        a different model for retrieval purposes, so the manifest binds the pair
        rather than each half.
        """
        return hashlib.sha256(
            canonical_json(
                {"query": self.query_prompt.payload(), "document": self.document_prompt.payload()}
            ).encode("utf-8")
        ).hexdigest()

    def prompt(self, *, kind: str) -> RetrievalPromptSpec:
        """The prompt for one direction, addressed by name."""
        if kind == "query":
            return self.query_prompt
        if kind == "document":
            return self.document_prompt
        raise BenchmarkContractError(
            f"retrieval prompt kind {kind!r} is not one of {list(RES138_PROMPT_NAMES)}. There are "
            "exactly two directions, and choosing a third would be inventing prompt policy.",
            operation="model_candidate_spec",
            model_id=self.model_id,
        )

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this candidate."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "trust_remote_code": self.trust_remote_code,
            "compute_dtype": self.compute_dtype,
            "output_dtype": self.output_dtype,
            "pooling_mode": self.pooling_mode,
            "native_max_sequence_length": self.native_max_sequence_length,
            "sequence_length_source": self.sequence_length_source,
            "query_prompt": self.query_prompt.payload(),
            "document_prompt": self.document_prompt.payload(),
            "prompt_sha256": self.prompt_sha256,
        }


RES138_SUPPORTED_DTYPES: Final[tuple[str, ...]] = ("float32",)
"""The only dtype names any RES-138 candidate may declare, for compute and for output.

**One value, on purpose, and not a tuning constant.** The TEI 1.9.4 reference on this
machine reported ``model_dtype float32``, and TEI equivalence is the gate that decides
whether native Colab vectors may be used for production at all — so float32 is the
reference-compatible choice. It is also the choice that keeps the four candidate runs
comparable with one another: two candidates computed in different precisions would not
be measuring the same thing, and a per-candidate precision is exactly the kind of
post-hoc freedom this table exists to remove.

Both models are small enough that float32 is expected to fit on a normal Colab GPU. **If
it does not, that is a feasibility finding to report, not a licence to switch.** Adding
``bfloat16`` here would be an amendment to the frozen contract, and the honest way to make
one is to edit this tuple, re-run the plan and publish a new plan digest — before any
quality number exists, and with the reason recorded.
"""


RES138_POOLING_MODES: Final[tuple[str, ...]] = ("mean", "last_token")
"""Pooling modes the runner will accept, matching what the two candidates declare.

Read from the loaded model's own pooling configuration and compared against the
frozen expectation. Recorded because it is not a request parameter: it is
decided when the model is built, and mean pooling versus last-token pooling over
identical weights produces vectors in different spaces entirely.
"""


def _prompt(name: str, content: str) -> RetrievalPromptSpec:
    return RetrievalPromptSpec(name=name, content=content, content_sha256=text_sha256(content))


RES138_MODEL_CANDIDATES: Final[tuple[ModelCandidateSpec, ...]] = (
    ModelCandidateSpec(
        model_id="voyageai/voyage-4-nano",
        revision="67fabc9bef010dabc5f6024aa1b1b6b93410426f",
        license="Apache-2.0",
        trust_remote_code=True,
        compute_dtype="float32",
        output_dtype="float32",
        pooling_mode="mean",
        native_max_sequence_length=32768,
        sequence_length_source="sentence_bert_config.json:max_seq_length at the pinned revision",
        query_prompt=_prompt("query", "Represent the query for retrieving supporting documents: "),
        document_prompt=_prompt("document", "Represent the document for retrieval: "),
    ),
    ModelCandidateSpec(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        license="Apache-2.0",
        trust_remote_code=False,
        compute_dtype="float32",
        output_dtype="float32",
        pooling_mode="last_token",
        native_max_sequence_length=32768,
        sequence_length_source=(
            "config.json:max_position_embeddings at the pinned revision; "
            "tokenizer_config.json declares 131072, which the model cannot positionally reach"
        ),
        query_prompt=_prompt(
            "query",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query"
            "\nQuery:",
        ),
        document_prompt=_prompt("document", ""),
    ),
)
"""The two frozen candidates, in a fixed order.

Both Apache-2.0, both publicly downloadable without a token, both 32K-context
Matryoshka models that support 512 and 1024. Prompt contents are transcribed
from ``config_sentence_transformers.json`` **at these revisions** — including the
trailing space on Voyage's two prompts and the absence of one after Qwen's
``Query:`` — and the runner re-reads that file and fails if the repository no
longer says what is frozen here.

A ``None`` default prompt and a ``cosine`` similarity function are declared by
both repositories, which is what makes ``prompt_name = "query"`` / ``"document"``
contractually available to TEI for both models.

**``trust_remote_code`` differs between the two, and that difference is the whole
reason it is a field.** Voyage 4 Nano ships custom modelling code, and its own
sentence-transformers usage requires the flag; the pinned repository cannot be
constructed without it. Qwen does not, so the flag is ``False`` for it. A runner
that branched on ``model_id`` would produce the same two answers today and would
be one candidate away from a third candidate inheriting the wrong answer, which is
why the policy lives in the frozen identity instead of in the code that loads it.

**Both compute and output dtypes are ``float32``**, for the reasons in
:data:`RES138_SUPPORTED_DTYPES`. Qwen's pinned ``config.json`` declares
``bfloat16``, and Voyage's recommended GPU path uses BF16; neither is inherited
here. The requested dtype is passed to ``SentenceTransformer`` explicitly, and the
dtype observed on the loaded parameters afterwards is what gets recorded.
"""

RES138_MODEL_IDS: Final[tuple[str, ...]] = tuple(
    candidate.model_id for candidate in RES138_MODEL_CANDIDATES
)
"""The frozen model ids, in the same order as the candidates."""


# ---------------------------------------------------------------------------
# Frozen numerics
# ---------------------------------------------------------------------------

RES138_RETRIEVAL_TOP_K: Final[int] = 100
"""Rankings retain 100 documents per query, always.

Enough for ``Recall@100`` to be informative: on the smallest frozen corpus
(NFCorpus, 3,633 documents) a top-100 cut is a real discrimination, and on the
largest (TREC-COVID, 171,331) it is a small sample. Retaining 100 also means
``nDCG@10`` is always computable from the retained set.
"""

RES138_NDCG_CUTOFF: Final[int] = 10
RES138_RECALL_CUTOFFS: Final[tuple[int, ...]] = (10, 100)

RES138_CORPUS_CHUNK_SIZE: Final[int] = 8192
"""Corpus rows scored per chunk.

A full query-by-corpus score matrix is never allocated. TREC-COVID with 50
queries over 171,331 documents would be 8.6 million float32 scores per dimension
— large enough to matter, and the reason a chunked scan exists at all. Chunking
is a memory decision only: it does not change a single score, because every
chunk is scored against the same query rows with the same arithmetic.
"""

RES138_SHARD_SIZE: Final[int] = 4096
"""Rows per embedding shard, frozen before any full model pass.

Sharding from the first shard rather than retrofitting it is what makes a
Colab session interruptible: a session that dies after 40 of 42 TREC-COVID
shards resumes with two shards of work instead of restarting three hours of it.
Small enough that the largest shard is a few seconds of GPU work, large enough
that 42 shards do not become 42 round-trips of per-shard overhead.

Changing it requires changing ``res138-shard-v4``, because the shard ordinal, the
row range and the ordered-id digest are all derived from it.
"""

RES138_RUN_ID_PREFIX: Final[str] = "colab"
"""Prefix of a Colab run identity, so a Drive run folder is recognisable as one."""

RES138_BOOTSTRAP_SEED: Final[int] = 138
RES138_BOOTSTRAP_SAMPLES: Final[int] = 10_000
RES138_BOOTSTRAP_CONFIDENCE: Final[float] = 0.95
"""The paired bootstrap, frozen.

Seed, replicate count and confidence level are constants because each of them
changes the conclusion while looking like an implementation detail. The
replicates are drawn from the standard library ``random.Random`` rather than a
third-party generator so the resampling stream does not depend on a library
version.
"""

RES138_CALIBRATION_BANDS: Final[tuple[str, ...]] = ("short", "typical", "long")
"""Length bands the calibration sample is drawn from, per workload and per kind.

**Within-workload** thirds, not absolute character thresholds, and that is the
whole point. Absolute bands cannot cover both a corpus whose shortest document is
4 characters and one whose shortest is 221, nor a query set whose longest query
is 72 characters and one whose longest is 204. Ranking each workload's own items
by length and taking the thirds guarantees every workload contributes to every
cell, which is what makes "multiple datasets" a property of the set rather than
a hope.
"""

RES138_CALIBRATION_ITEMS_PER_CELL: Final[int] = 2
RES138_CALIBRATION_TOP_K: Final[int] = 10
"""Calibration set shape: 2 items per (workload, kind, band) cell, top-10 ordering.

3 workloads x (queries, documents) x 3 bands x 2 = 36 items. Small enough that
four encode passes per model finish in minutes, large enough that a top-10
ordering comparison over 18 items per path has ties and near-ties to disagree
about.
"""


@dataclass(frozen=True)
class MrlCalibrationGate:
    """The predeclared tolerance that decides whether a derived vector may be used.

    Frozen *before* any calibration is run, and never adjusted afterwards: a
    tolerance read off the observed difference is not a gate.

    ``minimum_cosine`` is compared against the **worst** vector, not the mean. A
    mean would let 999 excellent vectors hide one derived vector that is off in
    the eleventh decimal, and that one vector is still a vector this benchmark
    would have measured quality with.
    """

    minimum_cosine: float
    maximum_absolute_difference: float
    require_identical_top_k: bool

    def __post_init__(self) -> None:
        for name, value, bound in (
            ("minimum_cosine", self.minimum_cosine, 0.0),
            ("maximum_absolute_difference", self.maximum_absolute_difference, 0.0),
        ):
            if not isfinite(value) or value < bound or value > 1.0:
                raise BenchmarkContractError(
                    f"MRL calibration gate {name} is {value!r}, which is not a finite value in "
                    f"[{bound}, 1]. A gate outside that range either rejects everything or "
                    "accepts a difference that is not a difference.",
                    operation="mrl_calibration_gate",
                )

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this gate."""
        return {
            "minimum_cosine": self.minimum_cosine,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "require_identical_top_k": self.require_identical_top_k,
        }


RES138_MRL_CALIBRATION_GATE: Final[MrlCalibrationGate] = MrlCalibrationGate(
    minimum_cosine=0.999999,
    maximum_absolute_difference=1e-5,
    require_identical_top_k=True,
)
"""Native-512 must equal derive-prefix-renormalise(native-1024), per model and path.

If a model/path pair fails any of the three conditions, ``derived512_allowed`` is
false for it and the full benchmark performs a **native 512 forward pass** for
that pair. The four evaluated configurations are the same either way; only the
number of GPU passes changes.
"""

# The production equivalence gate and the production TEI runtime are Stage B
# contracts and live in :mod:`dynamisrag.benchmark.production`. Stage A does not
# evaluate them: its reference vectors are the thing a production configuration
# must later be proven equivalent to, so Stage A cannot also be the thing that
# performs that proof.

RES138_ARTIFACT_REVISIONS: Final[Mapping[str, str]] = {
    "plan": "res138-plan-v3",
    "runtime": "res138-runtime-v1",
    "source_manifest": "res138-source-manifest-v1",
    "model_manifest": "res138-model-manifest-v1",
    "shard": "res138-shard-v4",
    "calibration_selection": "res138-calibration-selection-v1",
    "mrl_calibration": "res138-mrl-calibration-v1",
    "preflight": "res138-preflight-v5",
    "results": "res138-results-v1",
    "query_results": "res138-query-results-v1",
    "workload_metrics": "res138-workload-metrics-v1",
    "macro_metrics": "res138-macro-metrics-v1",
    "bootstrap": "res138-bootstrap-v1",
    "performance": "res138-performance-v4",
    "full_run": "res138-full-run-v3",
    "selection": "res138-selection-v2",
    "production_qualification": "res138-production-qualification-v1",
    "long_context": "res138-long-context-benchmark-v1",
}
"""Every artifact this benchmark writes, and the revision each one declares.

Named in one place because an artifact whose identity does not name its own
schema cannot be compared with, or replaced by, another one. A change to what
any artifact *binds* arrives as a new revision string here, never as a silent
difference in a payload.

The revision bumps of the staged amendment are deliberate and each answers a
schema or identity change: ``plan`` v3 declares the three stages, the 8192
reference boundary and the staged execution policy; ``preflight`` v5 carries
schedule-execution probes and the stage marker instead of A100 memory-eligibility
materials and TEI declarations; ``performance`` v4 marks Stage A execution
observations as reference measurements that are not production throughput;
``full-run`` v3 declares completed quality evidence and points production
qualification at Stage B; ``selection`` v2 admits operational metrics only from
Stage B. ``shard`` stays at v4 because the shard sidecar schema did not change —
the boundary and policy it binds are recomputed from the persisted raw counts,
so a pre-amendment sidecar is refused by the data, not by a revision string.

``calibration_selection`` is in this list rather than beside the other
calibration constants because it *is* an artifact: the provenance record of which
items were drawn and why. Its revision is an input to the item digest, so letting
it drift away from the table would let two builds draw different items while
agreeing on every revision they published.
"""

RES138_CALIBRATION_SELECTION_REVISION: Final[str] = RES138_ARTIFACT_REVISIONS[
    "calibration_selection"
]


# ---------------------------------------------------------------------------
# The Drive contract
#
# Paths are the notebook-facing contract. Folder ids are recorded for provenance
# and documentation only and are never required to compute anything: a run that
# needed a folder id to start would break the moment the tree is reorganised.
# ---------------------------------------------------------------------------

RES138_DRIVE_ROOT: Final[str] = "/content/drive/MyDrive/DynamisRAG/RES-138"
"""Mounted Drive root for this issue, after ``drive.mount("/content/drive")``."""

RES138_LOCAL_SCRATCH_ROOT: Final[str] = "/content/res138"
"""Where all active work happens.

Downloads, extraction, matrix generation, scoring and hashing run here, on the
Colab ephemeral disk. Drive holds finished checkpoints and evidence only, and is
never used for large random I/O: a mounted Drive is a network filesystem, and
writing a shard matrix into it byte by byte turns a compute run into an I/O
benchmark.
"""


@dataclass(frozen=True)
class DriveLocation:
    """One node of the Drive tree: a mounted path, and the id it was created with.

    ``folder_id`` is documentation. It exists so an operator can confirm *which*
    folder a run wrote to by opening Drive, and it is never an input to any
    computation.
    """

    label: str
    path: str
    folder_id: str | None = None


RES138_DRIVE_LOCATIONS: Final[tuple[DriveLocation, ...]] = (
    DriveLocation(
        "DynamisRAG", "/content/drive/MyDrive/DynamisRAG", "1SmQECt6EBtnYczxXCjlgkU1UnvGssbMl"
    ),
    DriveLocation("res-138", RES138_DRIVE_ROOT, "1KGaSK8zZCEgxbN4hH6dGesJ3svswNTMU"),
    DriveLocation(
        "notebooks", f"{RES138_DRIVE_ROOT}/notebooks", "1oqLjiuxo_8MaMopYHtoReJ8m83B-SobQ"
    ),
    DriveLocation("sources", f"{RES138_DRIVE_ROOT}/sources", "15D1eo2DlC8FW314A5XTS5iHBD30WGYmp"),
    DriveLocation(
        "sources/beir", f"{RES138_DRIVE_ROOT}/sources/beir", "1mzq7v8A3rMtyqqliB3fkkP-JQtLO3yuZ"
    ),
    DriveLocation("runs", f"{RES138_DRIVE_ROOT}/runs", "1ie9oB15ICyuwgyq4_thOGKOUaSWKzknU"),
)
"""The reference storage contract, as a flat list in tree order."""


def drive_path(*parts: str) -> str:
    """Join mounted paths under the Drive root, without touching the filesystem."""
    return "/".join((RES138_DRIVE_ROOT, *parts))


# ---------------------------------------------------------------------------
# Workload value types
# ---------------------------------------------------------------------------

_WORKLOAD_OPERATION: Final[str] = "retrieval_workload"


def beir_document_embedding_text(title: str, body: str) -> str:
    """The one definition of the string a BEIR document is embedded as.

    ``"<stripped title>\\n<body>"`` when the title has content, and ``"<body>"``
    alone when it does not. The newline is a field separator rather than a space
    because it cannot merge a title ending in punctuation with the first word of
    the body, and because it survives a body that already begins with a newline.

    The title is stripped and the body is **not**: no whitespace normalisation, no
    case folding, no de-duplication of blank lines. Those operations would change
    the tokens the model sees, and the content digest is taken over whatever this
    function returns, so a normalisation applied here would have to be declared
    here or nowhere.

    A trailing-space prompt and this separator are the reason the policy carries
    a revision: it is part of the vectors.
    """
    stripped = title.strip()
    if stripped:
        return f"{stripped}\n{body}"
    return body


@dataclass(frozen=True)
class RetrievalDocument:
    """One corpus document, frozen, content-addressed and in canonical order.

    ``text`` is **the string that is embedded**, not the raw BEIR body: the
    loader applies :func:`beir_document_embedding_text` first, so ``title`` is
    retained for audit and ``text`` is what the tokenizer receives. Keeping the
    two apart is what lets a reviewer confirm that the title was included
    without having to re-derive the join.

    ``content_sha256`` is verified against ``text`` on construction rather than
    checked for shape. A well-formed digest belonging to other text would let an
    artifact attest to content it never embedded.

    The document text is third-party scientific literature. It is never echoed
    by any error raised here or downstream.
    """

    document_id: str
    title: str
    text: str
    content_sha256: str

    def __post_init__(self) -> Self:
        if not self.document_id:
            raise BenchmarkContractError(
                "a retrieval document must carry a document_id; an unnamed corpus row cannot be "
                "ranked, ordered canonically, or joined to a qrel.",
                operation="retrieval_document",
            )
        if not self.text.strip():
            raise BenchmarkContractError(
                f"retrieval document {self.document_id!r} has no embeddable text. RES-137 refuses "
                "empty passage text for the same reason: a model returns a well-formed but "
                "meaningless vector for one, and such a vector would occupy a corpus slot and "
                "perturb Recall@100. The loader must exclude it and record the exclusion.",
                operation="retrieval_document",
                item_id=self.document_id,
            )
        observed = text_sha256(self.text)
        if observed != self.content_sha256:
            raise BenchmarkContractError(
                f"retrieval document {self.document_id!r} declares content_sha256 "
                f"{self.content_sha256!r} but SHA-256 over the exact UTF-8 bytes of its text is "
                f"{observed!r}. Two records of one fact that disagree would bind the artifact to "
                "text it never embedded. The document text is deliberately not reported.",
                operation="retrieval_document",
                item_id=self.document_id,
                expected=self.content_sha256,
                observed=observed,
            )
        return self

    @classmethod
    def from_beir(cls, *, document_id: str, title: str, body: str) -> Self:
        """Build from raw BEIR fields, applying the frozen document text policy."""
        text = beir_document_embedding_text(title, body)
        return cls(
            document_id=document_id,
            title=title.strip(),
            text=text,
            content_sha256=text_sha256(text),
        )

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this document. Never includes the text itself."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class RetrievalQuery:
    """One query, frozen and content-addressed.

    Query text is embedded verbatim: there is no join to apply, so there is no
    policy to declare. The content digest binds the exact bytes, for the same
    reason it does on a document.
    """

    query_id: str
    text: str
    content_sha256: str

    def __post_init__(self) -> Self:
        if not self.query_id:
            raise BenchmarkContractError(
                "a retrieval query must carry a query_id; an unnamed query cannot be joined to "
                "its qrels or to a metric row.",
                operation="retrieval_query",
            )
        if not self.text.strip():
            raise BenchmarkContractError(
                f"retrieval query {self.query_id!r} has no embeddable text, so it has no query "
                "vector and cannot be scored. The loader must exclude it and record the exclusion.",
                operation="retrieval_query",
                item_id=self.query_id,
            )
        observed = text_sha256(self.text)
        if observed != self.content_sha256:
            raise BenchmarkContractError(
                f"retrieval query {self.query_id!r} declares content_sha256 "
                f"{self.content_sha256!r} but SHA-256 over the exact UTF-8 bytes of its text is "
                f"{observed!r}. The digest binds the metric row to the query that produced it.",
                operation="retrieval_query",
                item_id=self.query_id,
                expected=self.content_sha256,
                observed=observed,
            )
        return self

    @classmethod
    def from_beir(cls, *, query_id: str, text: str) -> Self:
        """Build from a raw BEIR query record."""
        return cls(query_id=query_id, text=text, content_sha256=text_sha256(text))

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this query. Never includes the text itself."""
        return {"query_id": self.query_id, "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class RetrievalQrel:
    """One judgment: this query, that document, this relevance.

    ``relevance`` is an integer judgment level and **any** integer is structurally
    valid, including zero and the negative values TREC-COVID's ``test.tsv``
    contains. Metric semantics — gain is ``2**relevance - 1`` when
    ``relevance > 0`` and ``0`` otherwise — live in
    :mod:`dynamisrag.benchmark.metrics`, because they are a property of a metric
    rather than of a judgment file. Refusing a negative judgment here would
    refuse a real BEIR distribution.
    """

    query_id: str
    document_id: str
    relevance: int

    def __post_init__(self) -> None:
        if not self.query_id or not self.document_id:
            raise BenchmarkContractError(
                "a retrieval qrel must name both a query_id and a document_id; a judgment about "
                "half a pair cannot be scored.",
                operation="retrieval_qrel",
            )
        # `bool` is an `int` subclass, so `True` would be relevance 1 and would
        # serialise into the artifact as `true` where a judgment level is meant.
        # Any integer is structurally valid -- TREC-COVID's `test.tsv` carries
        # `-1` -- so the check is on the type, never on the sign or the range.
        object.__setattr__(
            self,
            "relevance",
            require_exact_int(
                self.relevance,
                kind="relevance judgment level",
                operation="retrieval_qrel",
                minimum=-(2**31),
                because="Any integer judgment level is structurally valid, including zero and "
                "the negative values TREC-COVID ships; the metric gain rule, not this contract, "
                "decides which levels count as relevant.",
            ),
        )

    @property
    def is_relevant(self) -> bool:
        """Whether this judgment counts as relevant, per the metric gain rule."""
        return self.relevance > 0

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this judgment."""
        return {
            "query_id": self.query_id,
            "document_id": self.document_id,
            "relevance": self.relevance,
        }


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    """SHA-256 over the canonical JSON of an ordered id list.

    This is the digest that says *which* rows a matrix holds. A matrix of 4,096
    float32 rows says nothing about which 4,096 documents they are, and a shard
    sidecar that recorded only a row count would let a shard file be replaced by
    another shard with the same shape and pass every size check.
    """
    return hashlib.sha256(canonical_json(list(ids)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrievalWorkload:
    """One frozen workload: corpus, queries and judgments, all in canonical order.

    Canonical order is enforced, not produced: ``documents`` ascend by
    ``document_id``, ``queries`` by ``query_id``, ``qrels`` by
    ``(query_id, document_id)``. A loader hands over sorted tuples; a
    hand-constructed workload with the right members in the wrong order is
    refused, because the order decides the request sequence and therefore the
    shard boundaries.

    Every qrel must name a query and a document this workload holds. A dangling
    reference is refused rather than dropped: dropping it would silently change
    the metric denominator for exactly the query it names.
    """

    name: str
    documents: tuple[RetrievalDocument, ...]
    queries: tuple[RetrievalQuery, ...]
    qrels: tuple[RetrievalQrel, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise BenchmarkContractError(
                "a retrieval workload must be named; results are reported per workload and an "
                "unnamed one could not be attributed to a source.",
                operation=_WORKLOAD_OPERATION,
            )
        if not self.documents:
            raise BenchmarkContractError(
                f"retrieval workload {self.name!r} holds no documents, so there is nothing to "
                "retrieve.",
                operation=_WORKLOAD_OPERATION,
                workload=self.name,
            )
        if not self.queries:
            raise BenchmarkContractError(
                f"retrieval workload {self.name!r} holds no queries, so nothing can be scored.",
                operation=_WORKLOAD_OPERATION,
                workload=self.name,
            )
        self._require_canonical_documents()
        self._require_canonical_queries()
        self._require_referenced_qrels()

    def _require_canonical_documents(self) -> None:
        ids = [document.document_id for document in self.documents]
        self._require_strictly_ascending(
            ids, kind="document_id", operation="retrieval_workload", workload=self.name
        )

    def _require_canonical_queries(self) -> None:
        ids = [query.query_id for query in self.queries]
        self._require_strictly_ascending(
            ids, kind="query_id", operation="retrieval_workload", workload=self.name
        )

    def _require_referenced_qrels(self) -> None:
        keys = [(qrel.query_id, qrel.document_id) for qrel in self.qrels]
        self._require_strictly_ascending(
            keys,
            kind="(query_id, document_id)",
            operation="retrieval_workload",
            workload=self.name,
        )
        query_ids = {query.query_id for query in self.queries}
        document_ids = {document.document_id for document in self.documents}
        for qrel in self.qrels:
            if qrel.query_id not in query_ids:
                raise BenchmarkContractError(
                    f"retrieval workload {self.name!r} holds a qrel for query {qrel.query_id!r}, "
                    "which it does not contain. Dropping the judgment would change the metric "
                    "denominator for exactly that query, so the dangling reference is refused.",
                    operation="retrieval_workload",
                    workload=self.name,
                    item_id=qrel.query_id,
                )
            if qrel.document_id not in document_ids:
                raise BenchmarkContractError(
                    f"retrieval workload {self.name!r} holds a qrel for document "
                    f"{qrel.document_id!r}, which it does not contain. Scoring it would compare a "
                    "ranking against a document that cannot appear in one.",
                    operation="retrieval_workload",
                    workload=self.name,
                    item_id=qrel.document_id,
                )

    @staticmethod
    def _require_strictly_ascending(
        keys: Sequence[str] | Sequence[tuple[str, str]],
        *,
        kind: str,
        operation: str,
        workload: str,
    ) -> None:
        """Require a strictly ascending sequence: sorted *and* without repeats.

        Strictly, not merely sorted, because a repeated key is not a tie to break
        — it is one row counted twice, and which of the two would win is an
        accident of iteration order.
        """
        previous: str | tuple[str, str] | None = None
        for current in keys:
            if previous is not None:
                if current == previous:
                    raise BenchmarkContractError(
                        f"retrieval workload {workload!r} repeats {kind} {current!r}. One row "
                        "counted twice makes the corpus, the query set or the judgment set "
                        "disagree with itself, and which copy would win is an accident of "
                        "iteration order.",
                        operation=operation,
                        workload=workload,
                        item_id=str(current),
                    )
                if not _ordered_before(previous, current):
                    raise BenchmarkContractError(
                        f"retrieval workload {workload!r} is not in canonical {kind} ascending "
                        f"order ({previous!r} precedes {current!r}). The canonical order decides "
                        "the request sequence the model saw and therefore the shard boundaries, so "
                        "it is enforced rather than produced by a sort in the middle of a run.",
                        operation=operation,
                        workload=workload,
                        item_id=str(current),
                    )
            previous = current

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Corpus ids in canonical order."""
        return tuple(document.document_id for document in self.documents)

    @property
    def query_ids(self) -> tuple[str, ...]:
        """Query ids in canonical order."""
        return tuple(query.query_id for query in self.queries)

    @property
    def qrels_by_query(self) -> Mapping[str, tuple[RetrievalQrel, ...]]:
        """Judgments grouped by query id, each group in canonical order."""
        grouped: dict[str, list[RetrievalQrel]] = {}
        for qrel in self.qrels:
            grouped.setdefault(qrel.query_id, []).append(qrel)
        return {query_id: tuple(group) for query_id, group in grouped.items()}

    @property
    def document_texts(self) -> tuple[str, ...]:
        """Every corpus document's embeddable text, in canonical order."""
        return tuple(document.text for document in self.documents)

    @property
    def query_texts(self) -> tuple[str, ...]:
        """Every query's embeddable text, in canonical order."""
        return tuple(query.text for query in self.queries)

    def summary(self) -> Mapping[str, object]:
        """Counts and ordered-id digests — never any text.

        Enough for a source manifest to state what a workload contains without
        republishing a corpus, and enough for two manifests to be compared for
        equality without holding either corpus.
        """
        return {
            "name": self.name,
            "document_count": len(self.documents),
            "query_count": len(self.queries),
            "qrel_count": len(self.qrels),
            "document_ids_sha256": ordered_ids_sha256(self.document_ids),
            "query_ids_sha256": ordered_ids_sha256(self.query_ids),
        }


def _ordered_before(previous: str | tuple[str, str], current: str | tuple[str, str]) -> bool:
    """Total ``<`` over the two key shapes canonical order compares.

    ``str`` for document and query ids, ``tuple[str, str]`` for qrel pairs. Both
    are totally ordered, and a mixed pair cannot arise from this class — a
    workload's qrel keys and its id keys are built in different methods — so the
    mixed case returns ``False`` and is reported as an ordering violation rather
    than quietly accepted.
    """
    if isinstance(previous, str) and isinstance(current, str):
        return previous < current
    if isinstance(previous, tuple) and isinstance(current, tuple):
        return previous < current
    return False


def require_candidate_dimension(dimension: int, *, operation: str) -> int:
    """Require a dimension that is one of the four frozen candidate dimensions.

    Not merely a positive integer: an unevaluated dimension would produce
    vectors nothing in this benchmark can interpret, and the pair it would have
    to be derived from is frozen at 1024.
    """
    if dimension in RES138_CANDIDATE_DIMENSIONS:
        return dimension
    raise BenchmarkContractError(
        f"candidate dimension {dimension!r} is not one of the frozen "
        f"{list(RES138_CANDIDATE_DIMENSIONS)}. This benchmark evaluates exactly those dimensions; "
        "a fourth would need a plan revision, and an unplanned dimension would make a result "
        "uncomparable with every other one.",
        operation=operation,
        expected=str(RES138_CANDIDATE_DIMENSIONS),
        observed=str(dimension),
    )


def require_shard_size(shard_size: int, *, operation: str) -> int:
    """Require the frozen shard size.

    Refused rather than clamped for the same reason a batch size is never
    clamped: the shard ordinal, the row range and the ordered-id digest are all
    derived from it, so silently using a different one would produce artifacts
    under the same revision.
    """
    if shard_size == RES138_SHARD_SIZE:
        return shard_size
    raise BenchmarkContractError(
        f"shard_size {shard_size!r} is not the frozen {RES138_SHARD_SIZE}. Shard ordinals, row "
        "ranges and ordered-id digests are derived from the shard size, so a different one would "
        "produce artifacts under the same revision that no other reader could reproduce.",
        operation=operation,
        expected=str(RES138_SHARD_SIZE),
        observed=str(shard_size),
    )
