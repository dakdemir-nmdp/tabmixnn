# Manuscript reconciliation — October 20, 2026

Starting point: `058e17f` on `revision/methods-2026-10-15`, plus six local
result JSONs from the updated ten-dataset runs. The October 19 source/table
commit had not included these result files, and the compiled PDF still
described the older evaluation.

## Changes

- Matched the abstract, benchmark descriptions, results and conclusion to the
  current five-regression/five-classification evaluation.
- Corrected sample sizes (180–7,584), group counts (18–316), the Toenail
  positive label and the Exam zero-score threshold.
- Explained that dataset selection was revised after inspecting an earlier
  evaluation and is exploratory. Removed unsupported explanations that
  attributed performance differences to particular mechanisms.
- Corrected singular-fit reporting (Train only on the primary split),
  split-partition counts, the ten calibration rows, their 0.082–0.731 slope
  range and the current Toenail underprediction example.
- Reported the five-split rankings consistently, including the classical
  mixed model's higher mean AUROC on all five unseen-group classification
  tasks. These rankings are descriptive, not significance tests.
- Replaced the validation script's fixed “90” message with computed pair,
  record and metric counts. Required a complete unique pair inventory,
  common protocol, exact execution-code hashes including R, and matching
  current source/data hashes.
- Replaced the obsolete Oxboys predictor fixture with Spruce, retaining a
  check that its occasion index is excluded and its random slope is specified.
  Added a regression test for duplicate pairs and changed R execution code.
- Updated reproduction instructions and clearly separated the older
  nine-dataset execution notes.
- Reviewed both branches and documented why no review-only runtime changes
  are imported into this benchmarked manuscript implementation. See
  `branch-review-2026-10-20.md`.

## Validation

The current artifact set contains 100 dataset/regime/split combinations,
930 fitted-model records and 3,240 metric values. In the preceding status
review every metric was reconstructed from cached public source data and
saved predictions at relative/absolute tolerance `1e-10`; the maximum absolute
difference was approximately `7.1e-15`. Source/code hashes, prediction rows
and the known/unseen-group split rules passed. All ten generated tables and
the sensitivity summary reproduced exactly.

After this revision, `summarize_split_sensitivity.py` again reconstructed
all 3,240 metrics and generated the two sensitivity tables, calibration
table and updated validation counts. `build_manuscript_tables.py`
regenerated the seven primary tables. No model was refitted and no benchmark
score was changed.

Full test suite: **138 passed, 17 expected warnings** about nonstandard KL
weighting in tests that deliberately use it. This replaces the pre-update
137-test status; the obsolete dataset test is repaired.

`latexmk -pdf -interaction=nonstopmode -halt-on-error
tabmixnn_methods.tex` completed successfully. The rebuilt PDF has 33 pages,
with no undefined references, missing citations or overfull boxes. One
pre-existing underfull paragraph remains in the continuous-survival section.
All pages were rendered for layout inspection; the abstract, dataset/setup,
results/tables and conclusion pages were inspected at higher resolution.
The split-sensitivity introduction was moved before its tables so its heading
has explanatory text on the same page. No clipping or overlapping content
was found.

The statistical prose changes cover the abstract, conclusion and benchmark
sections, with corrections to labels/counts, numerical comparisons,
selection disclosure and unsupported mechanistic interpretations. The style
check found no em dashes or blocked vocabulary; sentence-rhythm flags were
reviewed without rewriting unaffected mathematical or technical material.

## Evidence and remaining limitations

The six updated aggregate JSON files belong with this manuscript revision.
Public source rows and per-row predictions remain in ignored
`experiments/data/`; no source identifiers are added to versioned artifacts.
The earlier dated audits remain historical records.

Specialized mixed-effects ML competitors, external/survival evaluation and
predictive-interval calibration remain outside the evidence. Branch
integration is documented, not performed. This is an author-review draft;
no push or submission is part of this revision.
