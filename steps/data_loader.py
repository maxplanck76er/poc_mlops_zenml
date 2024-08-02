import pandas as pd
from typing_extensions import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_loader(
    bucket_uri: str, input_filename: str
) -> Annotated[pd.DataFrame, "data_loader::raw_dataset"]:
    """Dataset reader step.

    This step is responsible for loading the raw input dataset

    Args:
        bucket_uri: bucket containing the input file
        input_filename: input file name.

    Returns:
        The dataset artifact as Pandas DataFrame and name of target column.
    """
    df = pd.read_csv(f"{bucket_uri}/{input_filename}", sep=';')[['Date de Création', ' NB_LIGNES_COMMANDE']]
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset with {len(df)} rows loaded!")

    return df
