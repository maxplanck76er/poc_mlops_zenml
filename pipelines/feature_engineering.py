from steps import (
    data_loader,
    data_preprocessor,
    data_splitter,
)

from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def feature_engineering(
    bucket_uri: str,
    input_filename: str,
    test_size: int = 60
):
    """
    Feature engineering pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets.

    Args:
        bucket_uri: The bucket uri
        input_filename: the input file name
        test_size: Size of holdout time series for training

    Returns:
        The processed datasets (dataset_trn, dataset_tst).
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    logger.info("Entering feature_engineering pipeline")
    #raw_data = data_loader(bucket_uri="gs://poc-ctb-data", input_filename="raw_data_histo.csv")
    raw_data = data_loader(bucket_uri=bucket_uri, input_filename=input_filename)
    logger.info(raw_data)

    dataset = data_preprocessor(
        dataset=raw_data
    )

    dataset_trn, dataset_tst = data_splitter(
        dataset=dataset,
        test_size=test_size,
    )
    return dataset_trn, dataset_tst
