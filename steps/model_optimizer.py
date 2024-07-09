from typing import Optional

import pandas as pd

from utils import utils
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_optimizer(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    num_trials: Optional[int] = 50,
    horizon: Optional[int] = 60,
) -> dict:
    """Hyperparameter tuning.

    Args:
        dataset_trn: The preprocessed train dataset.
        dataset_tst: The preprocessed test dataset.
        num_trials: The number of optimization trials.
        horizon: The forecast horizon.

    Returns:
        The best params as dict.

    """

    # Optimize
    best_params = utils.optimize(dataset_trn, dataset_tst, num_trials=num_trials, horizon=horizon)

    return best_params
