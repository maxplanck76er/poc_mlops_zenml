from steps import (
    data_loader,
    inference_predict,
    data_preprocessor,
)

from zenml import get_pipeline_context, pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def inference(bucket_uri: str, input_filename: str, ):
    """
    Model inference pipeline.

    This is a pipeline that loads the inference data, processes it with
    the same preprocessing pipeline used in training, and runs inference
    with the trained model.

    Args:
        random_state: Random state for reproducibility.
        target: Name of target column in dataset.
    """
    # Get the production model artifact
    model = get_pipeline_context().model.get_artifact("prophet_forecasting")

    # Get the preprocess pipeline artifact associated with this version
    preprocess_pipeline = get_pipeline_context().model.get_artifact(
        "preprocess_pipeline"
    )

    # Link all the steps together by calling them and passing the output
    #  of one step as the input of the next step.
    raw_data = data_loader(bucket_uri="gs://poc-ctb-data", input_filename="raw_data_histo.csv")

    df_inference = data_preprocessor(
        dataset=raw_data
    )

    inference_predict(
        model=model,
        dataset_inf=df_inference,
    )
