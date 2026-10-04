#!/usr/bin/env python
from src.logging_conf import configure_logging
from src.pipeline.stages import stage_evaluate

if __name__ == "__main__":
    configure_logging()
    stage_evaluate(force=True)
