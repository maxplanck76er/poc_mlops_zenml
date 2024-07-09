import pandas as pd
from prophet import Prophet
from utils import utils
from typing_extensions import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
    model: Prophet,
    dataset_inf: pd.DataFrame,
    horizon: int = 60
) -> Annotated[pd.Series, "predictions"]:
    """Predictions step.

    Args:
        model: Trained model.
        dataset_inf: The inference dataset.
        horizon: The forecast horizon.

    Returns:
        The predictions as pandas series
    """
    # run prediction from memory
    future = utils.make_future(model, dataset_inf, periods=horizon)
    if model.params['growth'] == 'logistic':
        future['cap'] = dataset_inf['y'].max()
        future['floor'] = 0.0

    predictions = model.predict(future)['yhat']

    return predictions
