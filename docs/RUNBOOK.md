# Runbook

## What each drift alert means

| flag | meaning | action |
|---|---|---|
| PSI ≤ 0.2, `ok` | distribution is stable enough | no page |
| 0.2 < PSI ≤ 0.25, `warning` | material shift on that feature | review `reports/drift_report.html`; schedule a retrain in the next cycle |
| PSI > 0.25, `serious` | large temporal shift | retrain the affected head; do not promote if the temporal score is worse than the dummy baseline |
| Performance degradation | random-split metric minus (or vs) train-on-pre-2019 / eval-on-2019+ | this is the number that tells you how often the model would need retraining |

The split is the real `date_added` axis (pre-2019 vs 2019+), not a synthetic perturbation. Expect real degradation around the 2020 catalogue-growth break.

## Retraining triggers

Retrain a head when **any** of:

1. Feature PSI > 0.25 on a feature that head is allowed to use.
2. Target PSI > 0.25 on that head's label.
3. Temporal performance worse than the dummy baseline.
4. Feature schema hash changes (the loader will refuse the old model).

Command: `make pipeline` (or `python scripts/run_pipeline.py --from-stage train`).

## Who would be paged

In this repository there is no on-call rota. In a production setting the
**MLOps owner** is paged on `serious` drift or a failed promotion of the
incumbent. The analyst who owns the insight feed is notified on a
surviving-finding count of zero (the engine may have become over-conservative)
or a sudden jump (thresholds may have been relaxed).

## Rollback

The registry keeps the previous promoted pointer.

```
python -c "from src.config import load_config; from src.tracking.registry import rollback; print(rollback(load_config(), 'type'))"
```

Replace `'type'` with the head. Then restart the API. There is no
in-place mutation of the previous artifact directory.

## Promotion refused

`make promote` prints `Promotion refused: gate '<name>' failed — ...`.
Do not override the gate. Fix the candidate or accept the incumbent.
