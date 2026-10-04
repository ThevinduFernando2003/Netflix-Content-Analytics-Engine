# 3-minute demo script

1. **The success-metric problem (30s).** Open the README first paragraph and the dashboard landing panel. Say: this file has no views, no watch time, no scores. We refused to invent a success label. The strongest genuine signal is `years_to_platform` — acquisition lag — labelled a proxy everywhere, including the API body.

2. **The pipeline (40s).** Run `python scripts/run_pipeline.py` on an already-built tree. Point at the log line `stage_skipped_cache_hit`. Then `python scripts/run_pipeline.py --from-stage insights` to show resumability. Open `reports/validation_report.html`.

3. **The insight engine (40s).** Open `reports/insights.md`. Read the candidates-tested vs survivors sentence. Expand one finding and show effect size, adjusted p, and n inline. Mention the noise test and the causal-verb CI check.

4. **Live `/analyze` (30s).** `curl` the flagship endpoint with Squid Game. Point at `acquisition_priority.is_proxy: true` and the prediction interval.

5. **Models and drift (30s).** Open `reports/model_leaderboard.md` — dummy baselines are on the same table. Open `reports/drift_report.html` and read the temporal degradation number. That is how often you would retrain.

6. **Architecture (10s).** The mermaid in the README: DAG → feature store → registry → API → dashboard. No Airflow. No training at container startup.
