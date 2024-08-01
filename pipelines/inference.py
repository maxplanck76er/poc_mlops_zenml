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
        bucket_uri: The bucket uri
        input_filename: the input file name
    """

    # Get the production model artifact
    model = get_pipeline_context().model.get_artifact("prophet_forecaster")

    logger.info(f"Inference model: {model}")

    # Link all the steps together by calling them and passing the output
    #  of one step as the input of the next step.
    raw_data = data_loader(bucket_uri=bucket_uri, input_filename=input_filename)

    df_inference = data_preprocessor(
        dataset=raw_data
    )

    inference_predict(
        model=model,
        dataset_inf=df_inference,
    )
