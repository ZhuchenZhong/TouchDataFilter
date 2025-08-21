import logging

from pathlib import Path
import torch


logger = logging.getLogger()

PROJECT_ROOT        = Path(__file__).absolute().parent
DATA_ROOT           = PROJECT_ROOT / "data"
MODEL_ROOT          = PROJECT_ROOT / "models"
SENSITIVITY_ROOT    = PROJECT_ROOT / "results" / "sensitivity"
PREDICTED_ROOT      = PROJECT_ROOT / "results" / "predicted"


if __name__ == "__main__":
    # Example usage
    logger.debug(f"Project root: {PROJECT_ROOT} is {'exists' if PROJECT_ROOT.exists() else 'not exists'}")
    logger.debug(f"Data root: {DATA_ROOT} is {'exists' if DATA_ROOT.exists() else 'not exists'}")
    logger.debug(f"Model root: {MODEL_ROOT} is {'exists' if MODEL_ROOT.exists() else 'not exists'}")
    logger.debug(f"Sensitivity results root: {SENSITIVITY_ROOT} is {'exists' if SENSITIVITY_ROOT.exists() else 'not exists'}")
    logger.debug(f"Predicted results root: {PREDICTED_ROOT} is {'exists' if PREDICTED_ROOT.exists() else 'not exists'}")
