# Decisions

Non-obvious choices, one line of reasoning each.

- RandomizedSearchCV uses `n_iter: 8` rather than 30: four tree ensembles times 30 draws times 3 folds on 8,790 rows with a text SVD block does not finish inside the 15-minute laptop budget, and a pipeline nobody can reproduce is not a pipeline.
- Hand-rolled DAG orchestrator instead of Prefect or Airflow: a single-container project does not justify a scheduler server, and a local-process DAG with file-hash caching meets the idempotency contract.
- Local file-backed MLflow (`mlruns/`): no tracking server to operate, and every run is still queryable for the auto-generated leaderboard.
- Lightweight parquet feature store rather than Feast: one team, one dataset, one machine; Feast would add infra without adding a consumer.
- Shipped file wins over the 12-column Kaggle spec: the provided xlsx has 8,790 rows and no `cast`/`description`; fabricating those columns would be a data lie.
- `"Not Given"` treated as missing (empty string + indicator), never as a real director/country: it is a cleaned-file sentinel, not a name.
- `catalogue_prominence` keeps the spec weights and zeros the missing `n_cast` / `description_length` terms rather than silently renormalising: the index is weaker, and renormalising would hide that.
- Acquisition heads ban every `date_added`-derived column including `year_added`: `years_to_platform = year_added - release_year`, so leaving `year_added` in hands the model half the target.
- Head A bans duration, TV-prefixed genre tokens, and `listed_in` itself: after the TV-token ban the remaining genre strings ("Movies", "Documentaries", empty) still separate type at ROC-AUC ≈ 1.00, so the head uses title text only. Serving builds that same title-only blob; concatenating `listed_in` at request time would put the banned signal back in.
- Cluster scoring keeps the fitted TF-IDF/SVD projector with the k-means model. A new title is embedded in that space; genre-mode assignment is only a fallback and does not invent a distance of zero.
- Promotion records store registry paths relative to the repo root so a container on another machine can still open the incumbent.
- Model `log1p(years_to_platform - floor)` and invert for metrics: the lag is right-skewed; MAPE is not reported because true values are frequently zero.
- Clustering uses L2-normalised TruncatedSVD + K-Means (spherical) and chooses *k* by silhouette **and** bootstrap ARI: Euclidean k-means on raw TF-IDF and elbow-only *k* are both misleading.
- Head F always reports a seasonal-naive baseline and prediction intervals: ~100 monthly points with a 2020 break cannot support a confident long-horizon point forecast.
- Insight engine requires BH-FDR **and** an effect-size floor: n = 8,790 makes trivial differences significant at α = 0.05.
- Definitional comparisons are not candidates: type vs `is_movie`, and acquisition bucket vs `years_to_platform` or `is_likely_original`, are true by construction.
- The train-stage cache hashes the feature file without `cluster_id` and `portfolio_fit`: those two columns are written by training into the same parquet, so hashing them made every later run look like a new input.
- Mann-Whitney U is the default two-group test: catalogue metrics are skewed; the code comments record the normality check that justified the default.
- Causal-verb stoplist is a CI-failing test: generated prose must not imply that metadata causes outcomes.
- SHAP / UMAP / Evidently stay out of the runtime image: they bloat the serve layer past the 2 GB budget and are precomputed at pipeline time.
- Pandera is imported only when the validation schema is built. The API loads the feature store, so `pyarrow` is on the runtime image and Pandera is not.
- Promotion is a coded rule with named gates: a human "this looks better" promotion is not reproducible.
- FastAPI lifespan loads artifacts once; the Streamlit app talks HTTP only: the dashboard must not import models, so serving and presentation stay independently deployable.
- Docker multi-stage copies `registry/` and `artifacts/` from a builder that already ran the pipeline: free-tier containers will time out if they train at startup.
- Optional TMDB enrichment is env-flagged and disk-cached: CI must never make a network call, and the pipeline must pass with the flag off.
