# TabMixNN revised manuscript (author-review draft)

The main source is `tabmixnn_methods.tex`, with `benchmarks_template.tex`,
`references.bib`, seven generated table inputs and `shap_examples.tex`.
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
The manuscript's results are generated from the three `manuscript_*.json`
aggregate files in `../experiments/results/`.

The downloaded arXiv archive and unused legacy `summary_table.tex`,
`regression_table.tex`, and `classification_table.tex` may remain locally.
They are not inputs to this revision and are not part of its versioned sources.
The current manuscript excludes performance claims unsupported by the new
nine-dataset evaluation. It is a draft for author review, not a submission.
