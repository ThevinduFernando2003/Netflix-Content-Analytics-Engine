#!/usr/bin/env python
from src.logging_conf import configure_logging
from src.pipeline.stages import stage_features

if __name__ == "__main__":
    configure_logging()
    stage_features(force=True)
