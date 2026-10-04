# Model card — Netflix catalogue analytics suite

This document covers every promoted head. It restates the §2 limitation
in full: **this platform analyses catalogue composition and acquisition
behaviour, not audience success.** The listing has no views, watch time,
completion, revenue, subscriber impact, IMDb/RT score, awards, budget or
renewal status. Success is not measurable from this file.

## Intended use

- Fill incomplete incoming listings (type, maturity ladder).
- Estimate **acquisition lag** (how long after release a title was added).
- Segment the catalogue for gap / duplicate stories.
- Forecast near-term monthly add volume, with a seasonal-naive baseline.

## Out-of-scope uses

- Predicting watch time, completion, revenue, or “what will be a hit”.
- Treating `is_likely_original` as a Netflix Originals ground-truth flag.
- Treating `catalogue_prominence` as an observed editorial or audience outcome.
- Long-horizon add-volume forecasts past a few months (2020 structural break).

## Training data and its defects

Shipped file: 8,790 rows, 10 columns. Missing `cast` and `description`
relative to the public 8,807-row Kaggle listing. `"Not Given"` is a
cleaned-file sentinel for director (2,588) and country (287), treated as
missing. `date_added` arrives as `M/D/YYYY`. Fourteen titles have
negative `years_to_platform` (not clipped to zero). `NR`/`UR` are dropped
from the ordinal maturity ladder.

## Evaluation data

A stratified 15% test set is held out once per supervised head. Head F
uses a time-based cutoff (default 2019-12). Metrics are written by
`src/models/evaluate.py` to `reports/model_leaderboard.md`.

## Metrics by head

Promoted heads have a card each under `docs/model_cards/`. The numbers in those
cards are copied from the promotion record by `src/tracking/cards.py`. See also
the auto-generated leaderboard. Ordinal heads report QWK, ordinal
MAE, adjacent accuracy, and the Kids-predicted-as-Mature count.
Regression reports MAE / RMSE / R² on the original scale; MAPE is
recorded only to demonstrate its failure near zero. Clustering reports
silhouette and bootstrap ARI. Forecasting always reports the
seasonal-naive baseline.

## Ethical considerations

- Maturity mis-ordering is a safety issue: Kids predicted as Mature is
  counted explicitly.
- Country and genre segments can encode regional production bias; the
  insight engine reports associations, never causes.
- An external ratings join would inherit TMDB/OMDb selection bias
  (non-English and older titles miss more often).

## Limitations (restated)

Proxy targets measure platform intent and metadata richness. They do not
measure audience success. Enabling `ENRICHMENT_ENABLED` is the honest
upgrade path; until then, do not put these scores on a “success” slide.
