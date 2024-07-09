from typing import Optional

import pandas as pd
from prophet import Prophet
from utils import utils

from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_evaluator(
    model: Prophet,
    best_params: dict,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    min_test_accuracy: float = 0.5,
    target: Optional[str] = "y",
    horizon: Optional[int] = 60
) -> float:
    """Evaluate a trained model.

    Args:
        model: The pre-trained model artifact.
        best_params: The optimized parameters.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        min_train_accuracy: Minimal acceptable training accuracy value.
        min_test_accuracy: Minimal acceptable testing accuracy value.
        target: Name of target column in dataset.
        horizon: The forecast horizon.

    Returns:
        The model accuracy on the test set.
    """
    # Calculate the model accuracy on the test set
    future = utils.make_future(model, dataset_trn, periods=horizon)
    if best_params['growth'] == 'logistic':
        future['cap'] = dataset_trn['y'].max()
        future['floor'] = 0.0

    forecast = model.predict(future)
    tst_acc = utils.compute_mase(dataset_trn[target], dataset_tst[target], forecast['yhat'])
    logger.info(f"Test accuracy={tst_acc}")

    messages = []
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc:.2f}% is below {min_test_accuracy:.2f}% !"
        )
    else:
        for message in messages:
            logger.warning(message)

    log_artifact_metadata(
        metadata={
            "test_accuracy": float(tst_acc),
        },
        artifact_name="prophet_forecaster",
    )
    return float(tst_acc)
