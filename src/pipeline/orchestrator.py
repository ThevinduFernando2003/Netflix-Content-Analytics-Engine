"""Plain-Python DAG. Idempotent, cached, resumable. No Airflow, no Prefect server."""

from __future__ import annotations

import argparse
from typing import Any

from src.config import load_config
from src.logging_conf import configure_logging, get_logger
from src.pipeline.stages import STAGE_FNS, STAGE_ORDER

log = get_logger(__name__)


def run_pipeline(
    from_stage: str | None = None,
    only: str | None = None,
    force: bool = False,
    fast: bool = False,
) -> list[dict[str, Any]]:
    cfg = load_config()
    if fast:
        cfg.settings.fast_mode = True
    if only:
        if only not in STAGE_FNS:
            raise ValueError(f"Unknown stage '{only}'. Known: {STAGE_ORDER}")
        return [STAGE_FNS[only](cfg, force=force)]

    start_idx = 0
    if from_stage:
        if from_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage '{from_stage}'. Known: {STAGE_ORDER}")
        start_idx = STAGE_ORDER.index(from_stage)
        force = True

    results: list[dict[str, Any]] = []
    for name in STAGE_ORDER[start_idx:]:
        # --from-stage forces the named stage and everything after; earlier stages stay cached.
        stage_force = force if from_stage is None else True
        results.append(STAGE_FNS[name](cfg, force=stage_force))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Netflix catalogue analytics pipeline")
    parser.add_argument("--from-stage", dest="from_stage", default=None)
    parser.add_argument("--only", dest="only", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fast", action="store_true", help="Shrink search spaces for tests / laptops"
    )
    args = parser.parse_args()
    configure_logging()
    results = run_pipeline(
        from_stage=args.from_stage,
        only=args.only,
        force=args.force,
        fast=args.fast,
    )
    skipped = sum(1 for r in results if r.get("skipped"))
    log.info("pipeline_complete", n_stages=len(results), n_skipped=skipped)


if __name__ == "__main__":
    main()
