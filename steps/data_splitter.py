from typing import Tuple

import pandas as pd
from typing_extensions import Annotated

from zenml import step


@step
def data_splitter(
    dataset: pd.DataFrame, test_size: int = 60
) -> Tuple[
    Annotated[pd.DataFrame, "data_splitter::dataset_trn"],
    Annotated[pd.DataFrame, "data_splitter::dataset_tst"],
]:
    """Dataset splitter step.

    Args:
        dataset: Processed dataset.
        test_size: defining portion of test set.

    Returns:
        The split dataset: dataset_trn, dataset_tst.
    """
    end_train = str(dataset['ds'].unique()[-test_size - 1]).split()[0]

    dataset_tst = dataset.loc[dataset['ds'] > end_train]
    dataset_trn = dataset.loc[dataset['ds'] <= end_train]

    return dataset_trn, dataset_tst