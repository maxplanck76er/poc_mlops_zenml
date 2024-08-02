import pandas as pd
from typing_extensions import Annotated

from zenml import step


@step
def inference_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame
) -> Annotated[pd.DataFrame, "inference_preprocessor:inference_dataset"]:
    """Data preprocessor step.

    Args:
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.

    Returns:
        The processed dataframe: dataset_inf.
    """

    dataset_inf = pd.concat([dataset_trn, dataset_tst]).reset_index(drop=True)

    return dataset_inf
