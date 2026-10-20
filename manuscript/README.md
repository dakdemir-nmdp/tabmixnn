# TabMixNN revised manuscript (author-review draft)

The main source is `tabmixnn_methods.tex`, with `benchmarks_template.tex`,
`references.bib`, ten generated table inputs, `shap_examples.tex` and `variational_appendix.tex`.
The three figure filenames retain their historical `shap_` prefix for source
compatibility; all plotted quantities and captions are feature ablations.
Vector PDF assets are used by LaTeX; PNG exports are also provided.

Build from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error tabmixnn_methods.tex
bibtex tabmixnn_methods
pdflatex -interaction=nonstopmode -halt-on-error tabmixnn_methods.tex
pdflatex -interaction=nonstopmode -halt-on-error tabmixnn_methods.tex
```

Experiment reproduction is documented in `../experiments/README.md`.
`../docs/manuscript-revision-2026-10-16.md` records the audit, changes and checks.
Primary results, four additional split runs, calibration/sensitivity summaries
and figure provenance are saved as `manuscript_*.json` in
`../experiments/results/`. The finite-dimensional symbolic check is stored
separately as `variational_identities.json`. The deep revision audit is
`../docs/deep-revision-2026-10-16.md`.

The downloaded arXiv archive and unused legacy `summary_table.tex`,
`regression_table.tex`, and `classification_table.tex` may remain locally.
They are not inputs to this revision and are not part of its versioned sources.
The current manuscript excludes performance claims unsupported by the new
ten-dataset evaluation (five regression and five classification tasks, five
splits in each of two prediction regimes). It is a draft for author review,
not a submission.

The October 20 reconciliation is documented in
`../docs/manuscript-reconciliation-2026-10-20.md`. It supersedes the older
nine-dataset counts and results in the dated audit reports. Use this branch's
canonical benchmark artifacts, not the different CV results on
`review/2026-10-14`. The branch comparison and integration disposition are in
`../docs/branch-review-2026-10-20.md`.
