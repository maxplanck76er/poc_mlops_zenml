from typing import Optional

import pandas as pd

from typing_extensions import Annotated
from prophet import Prophet
from utils import utils
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_trainer(
    dataset_trn: pd.DataFrame,
    best_params: Optional[dict],
    target: Optional[str] = "y"
) -> Annotated[
    Prophet,
    ArtifactConfig(name="prophet_forecaster", is_model_artifact=True),
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        best_params: The optimized parameters
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    # Training
    num_non_nan_rows = len(dataset_trn['y']) - dataset_trn['y'].isna().sum()
    logger.info(f"Missing values: {num_non_nan_rows != len(dataset_trn[target])}")

    m = utils.p_model(best_params, dataset_trn)
    train = utils.p_model_df(dataset_trn)
    if best_params['growth'] == 'logistic':
        train['cap'] = train['y'].max()
        train['floor'] = 0.0

    m.fit(train)

    return m
