import pandas as pd
from typing_extensions import Annotated

from zenml import step


@step
def data_preprocessor(
    dataset: pd.DataFrame
) -> Annotated[pd.DataFrame, "data_preprocessor::preprocessed_dataset"]:
    """Data preprocessor step.

    Args:
        dataset: The dataset.

    Returns:
        The processed dataset.
    """

    dataset = dataset.rename(columns={'Date de Création': 'ds', ' NB_LIGNES_COMMANDE': 'y'})
    dataset['ds'] = pd.to_datetime(dataset['ds'], format='mixed')

    # Filling the gaps
    global_start_date = dataset.reset_index()['ds'].min()
    global_end_date = dataset.reset_index()['ds'].max()
    print(f"date range : {global_start_date} - {global_end_date}")

    x = dataset.reset_index(drop=True).set_index('ds')
    # This is the complete date range (1 day frequency) from start_date to end_date
    dr = pd.date_range(start=global_start_date, end=global_end_date, freq="1D")
    x = x.reindex(dr)
    x = x.rename_axis('ds')
    # Fill generated NaN with 0, i.e. num_lines = 0 at each missing date
    x.fillna(0, inplace=True)
    dataset = x.reset_index()

    return dataset
