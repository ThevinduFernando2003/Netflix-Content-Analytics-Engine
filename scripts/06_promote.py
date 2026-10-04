#!/usr/bin/env python
from src.logging_conf import configure_logging
from src.pipeline.stages import stage_promote

if __name__ == "__main__":
    configure_logging()
    stage_promote(force=True)
