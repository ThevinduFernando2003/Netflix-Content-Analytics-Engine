#!/usr/bin/env python
from src.logging_conf import configure_logging
from src.pipeline.stages import stage_train

if __name__ == "__main__":
    configure_logging()
    stage_train(force=True)
