from typing import Optional
from uuid import UUID

from steps import model_evaluator, model_promoter, model_trainer, model_optimizer

from pipelines import (
    feature_engineering,
)
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def training(
    train_dataset_id: Optional[UUID] = None,
    test_dataset_id: Optional[UUID] = None,
    target: Optional[str] = "y"
):
    """
    Model training pipeline.

    This is a pipeline that loads the data from a preprocessing pipeline,
    trains a model on it and evaluates the model. If it is the first model
    to be trained, it will be promoted to production. If not, it will be
    promoted only if it has a lower mase than the current production
    model version.

    Args:
        train_dataset_id: ID of the train dataset produced by feature engineering.
        test_dataset_id: ID of the test dataset produced by feature engineering.
        target: Name of target column in dataset.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.

    # Execute Feature Engineering Pipeline
    if train_dataset_id is None or test_dataset_id is None:
        dataset_trn, dataset_tst = feature_engineering()
    else:
        client = Client()
        dataset_trn = client.get_artifact_version(
            name_id_or_prefix=train_dataset_id
        )
        dataset_tst = client.get_artifact_version(
            name_id_or_prefix=test_dataset_id
        )

    best_params = model_optimizer(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst
    )

    model = model_trainer(
        dataset_trn=dataset_trn, best_params=best_params, target=target
    )

    acc = model_evaluator(
        model=model,
        best_params=best_params,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
    )

    model_promoter(mase=acc)
