# Model card — Content type classification

This platform analyses catalogue composition and acquisition behaviour, not audience success.
The listing has no views, watch time, completion, revenue, subscriber impact, ratings scores,
awards, budget, or renewal status. The score from this head is a proxy.

- Head: `type`
- Task: `binary_classification`
- Primary metric: `roc_auc`
- Promoted: `True`
- Run id: `r1`
- Candidate: `logreg`

## Holdout metrics

- `roc_auc`: 0.9000
- `brier_after`: 0.1000
- `latency_ms_p95`: 5

