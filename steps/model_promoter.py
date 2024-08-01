from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(mase: float, stage: str = "production") -> bool:
    """Model promoter step.

    This step checks whether the model should be promoted to the next stage. If there is
    already a model in the indicated stage, the model with the best accuracy
    is promoted.

    Args:
        mase: Accuracy of the model.
        stage: Which stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    is_promoted = False

    if mase > 0.8:
        logger.info(
            f"Model mase {mase:.2f}% is above 0.8 ! Not promoting model."
        )
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True

        # Get the model in the current context
        current_model = get_step_context().model

        # Get the model that is in the production stage
        client = Client()
        try:
            stage_model = client.get_model_version(current_model.name, stage)
            # We compare their metrics
            prod_mase = (
                stage_model.get_artifact("prophet_forecaster")
                .run_metadata["test_accuracy"]
                .value
            )
            if float(mase) < float(prod_mase):
                # If current model has better metrics, we promote it
                is_promoted = True
                current_model.set_stage(stage, force=True)
        except KeyError:
            # If no such model exists, current one is promoted
            is_promoted = True
            current_model.set_stage(stage, force=True)
    return is_promoted
