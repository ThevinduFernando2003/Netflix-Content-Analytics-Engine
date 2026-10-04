from __future__ import annotations

from src.data.load import load_raw
from src.features.engineer import engineer_features
from src.models.clustering import fit_clusters


def test_two_feature_builds_identical(cfg):
    raw = load_raw(cfg.path("raw_csv"), cfg)
    a = engineer_features(raw, cfg)
    b = engineer_features(raw, cfg)
    assert a["catalogue_prominence"].tolist() == b["catalogue_prominence"].tolist()
    assert (
        a["years_to_platform"].fillna(-999).tolist() == b["years_to_platform"].fillna(-999).tolist()
    )


def test_clustering_labels_reproducible(featured, cfg):
    # Shrink the k grid for the unit test.
    cfg.models["heads"]["cluster"]["k_min"] = 3
    cfg.models["heads"]["cluster"]["k_max"] = 3
    cfg.models["heads"]["cluster"]["bootstrap_resamples"] = 4
    a = fit_clusters(featured, cfg)
    b = fit_clusters(featured, cfg)
    assert a["labels"].tolist() == b["labels"].tolist()
