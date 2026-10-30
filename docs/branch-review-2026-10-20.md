# Branch review and integration disposition — October 20, 2026

The manuscript revision uses `revision/methods-2026-10-15`, starting this
reconciliation at `058e17f`. The comparison branch is `review/2026-10-14`
at `bd68156`. They have 14 manuscript-only and 10 review-only commits.
The manuscript branch is the source of the current ten-dataset results.
This review does not merge the branches or mark their features equivalent.

## Disposition

Keep the tested manuscript implementation and its explicit experimental
protocol for this revision. No review-only runtime changes are required to
repair the reporting defects. Importing different inference or training
behavior would define a new experiment and require separate validation.
The branches remain available for later feature integration.

| Review-only work | Decision and reason |
|---|---|
| `687c541`, `bf7deaf`: `predict_mc_prior` and its optimization | Do not import in this revision. The manuscript explicitly evaluates zero-effect plug-in predictions for new groups; Monte Carlo marginal prediction is a different estimand. The proposed implementation also needs compatibility review for the repaired family interfaces and covariance transformations before use. No marginal-prediction superiority is claimed. |
| `2244040`: `GroupBatchSampler` | Do not import. A direct reproduction below shows that the implementation splits groups despite its stated contract. The manuscript uses full batches and does not need a cluster sampler. |
| `2244040`: `predict_interval` | Do not import. Its draws omit observation noise and weight/covariance-parameter uncertainty; they do not supply calibrated predictive intervals. The manuscript already states these limitations. Adding this interface would not close the predictive-coverage evidence gap. |
| `2244040`: variance/ICC caveat | The manuscript already includes latent-scale nonidentifiability and a counterexample. A response-scale predictive decomposition is not interchangeable with a latent-variance ratio. Do not copy the older branch's caveat as if the implementations were identical. |
| `687c541`: autoregressive/causal wording | The manuscript branch already explains the fixed triangular ordering and absence of causal identification. It also retains later corrections to gated sequential and hybrid fixed-point descriptions. No earlier docstring needs to replace those corrections. |
| `0dd2d3d`: random-slope default and dataset-specific embedding dimensions | The canonical benchmark explicitly requests UN covariance and embedding dimension 8. Preserve these recorded settings. Changing default behavior or dataset-specific dimensions is a separate model comparison. |
| `f29d2af`, `7c163cf`, `853dd19`, `f031d55`, `f4e508f`, `bd68156`: CV configuration, datasets, reruns and result documentation | Keep separate from the manuscript suite. The CV runs have different datasets, predictors and training settings. Their 3/6 and 3/5 win counts must not replace the manuscript's five-split comparisons. |

The manuscript-only changes include likelihood/family repairs, training and
refit corrections, prediction preprocessing, explicit mathematical limitations,
and a reproducible matched-comparator protocol. A wholesale merge from the
review branch would not be a substitute for porting selected features onto
these repaired interfaces.

## Reproduced sampler defect

The class definition was loaded directly from the review branch's checked-out
`core/model.py`, without editing either branch. With:

```python
group_indices = [1, 1, 1, 2, 2, 2, 3, 3, 3]
batch_size = 4
shuffle = False
```

the returned row-index batches are:

```text
[[0, 1, 2, 3], [4, 5, 6, 7], [8]]
```

Group 2 occurs in batches 0 and 1; group 3 occurs in batches 1 and 2.
`yield batch[:self.batch_size]` truncates a group at the target size.
This is sufficient reason not to import it as a working whole-group sampler.
The review branch itself has not been modified.

## Scope remaining

A future integration should validate each selected feature against the
manuscript branch's likelihood, output-shape, covariance and preprocessing
contracts, then rerun any benchmark whose training or prediction behavior
changes. This is proposed follow-up work, not a completed merge.
