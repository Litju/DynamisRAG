# DynamisBench v0 scientific question and review protocol

The 13 items in `evidence/dynamisbench-v0/candidate-bank.json` are proposals derived
from retained original JATS and canonical paragraphs. They are not expert ground
truth. The Methods, Results, tables and limitations were inspected locally; raw
text is retained privately. Every item records study identifiers, passage keys,
source anchors, span offsets/hashes, expected answerability, a proposed answer,
hard negatives where defensible, population/intervention restrictions, limitations
and human review requirements. Whole-passage candidate spans are deliberately
broad and must be narrowed during expert annotation.

| Item | Category | Source / scope |
|---|---|---|
| db00-01 | Direct factual | PMC10780924: 60° versus 90° muscle activity |
| db00-02 | Methodological | PMC10780924: healthy participants, load, dynamometer, sEMG |
| db00-03 | Numerical | PMC5263151: impulse/contact-time means, SD, units, p-values |
| db00-04 | Comparative | PMC5263151: tendon stiffness versus static torque |
| db00-05 | Multi-study | PMC10780924 / PMC10680285: attributed design and population comparison |
| db00-06 | Contradictory | PMC10680285: conflicting Results prose and Table 3 LCJ comparisons |
| db00-07 | Temporal | PMC10680285 / PMC5263151: duration, frequency and rest interval |
| db00-08 | Unanswerable | No five-year rupture-incidence evidence in this corpus |
| db00-09 | False premise | PMC10780924 does not establish long-term injury prevention in older patients |
| db00-10 | Methodological | PMC5263151: ultrasound and muscle-specific/generalization limitations |
| db00-11 | Numerical | PMC10680285 Table 3: PTG stationary vertical jump, baseline imbalance |
| db00-12 | False premise | PMC5263151: distinguish 21 recruited from 8/9 kinematic samples |
| db00-13 | Unanswerable | No optimal-dose evidence for pregnant elite marathon runners |

The contradictory item is an **internal reporting discrepancy**, not evidence of
disagreement between independent trials. Original Table 3 was inspected with its
column headers: LCJ pairwise CG/CTG, CTG/PTG, PTG/CG p-values are 1.000, 0.318,
0.038. Results prose first names CG/PTG but later names only CG/CTG. A reviewer
must confirm this reading and decide how to score a faithful uncertainty-aware
answer. The benchmark must not silently repair the paper or manufacture a
cross-study conflict. Multi-study coverage supports design comparison only.

Two independent sports-science reviewers should annotate each item before seeing
candidate-system comparisons. For each, inspect the original source (including
table structure and limitations), independently judge answerability, then label
pooled passages using 0/1/2/3. Retain unknown/unjudged as unknown. Check duplicate
chunks, front matter, hypotheses and cited third-party findings. Resolve conflicts
through documented adjudication rather than majority-by-automation.

For numerical evidence, record the estimand, group, condition/timepoint, units,
mean/SD versus SE/CI, direction, p-value, denominator and analysis population.
Do not infer an effect size from group summaries or equate non-significance with
equivalence. The cricket trial has baseline stationary-jump imbalance. The
depth-jump study has missing ankle-angle data and a small sample. The knee-angle
experiment is acute and cross-sectional; its outcomes are not longitudinal injury
or clinical efficacy evidence. Preserve these distinctions in scientific labels.

An exact quotation may still be irrelevant, a hypothesis, a background citation,
or an overgeneralized conclusion. Evaluate citation support and scientific
entailment separately from quotation membership. For false-premise items require
correction or justified abstention; a supported status alone is insufficient.
For corpus-unanswerable items require justified abstention; unrelated retrieved
passages must not become a fabricated negative relevance pool.

Each review must bind the canonical specification digest and, for answer review,
the question and exact full answer digest. Retain reviewer identity, timestamp,
protocol, rationale and externally verifiable signature. Keep signing authority
outside benchmark artifacts. Hashes or an automated process cannot approve a
scientific gold set. Freeze the adjudicated bank, pool and acceptance protocol
before evaluating a candidate for any future performance gate.
