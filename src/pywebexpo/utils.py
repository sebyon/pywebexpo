import numpy as np
from .model import WebExpoModel


def check_lognormal_data(data: np.ndarray, oel: float) -> bool:
    """
    Return True if all values are positive and within [oel/1000, 1000*oel].

    Args:
        data: Data to check.
        oel: Occupational Exposure Limit.
    """
    return np.all((data > 0) & (data >= oel / 1000) & (data <= 1000 * oel))


def check_normal_data(data: np.ndarray) -> bool:
    """
    Return True if all values are within the normal model range [40, 140].

    Args:
        data: Data to check.
    """
    return np.all((data >= 40) & (data <= 140))


def inspect_divergent_traces(
    model: WebExpoModel,
    warning_threshold: int = 1,
    critical_threshold: int = 10,
) -> dict[str, str | int]:
    """
    Inspect divergent transitions in a fitted WebExpoModel.

    Parameters
    ----------
    model : WebExpoModel
        A fitted WebExpoModel instance.
    warning_threshold : int
        Divergence percentage above which a warning is issued.
    critical_threshold : int
        Divergence percentage above which a critical warning is issued.

    Returns
    -------
    dict[str, str | int]
        A dictionary with a 'message' and 'divergent_transitions' count.
    """
    if model.idata.sample_stats is None:
        raise ValueError("The model has not been fit yet. Please run the fit method first.")

    diverging = model.idata.sample_stats["diverging"].values.flatten()
    diverging_count = np.count_nonzero(diverging)
    diverging_percentage = (diverging_count / diverging.size) * 100

    if diverging_count == 0:
        return {"message": "No divergent transitions detected.", "divergent_transitions": 0}

    if diverging_percentage >= critical_threshold:
        return {
            "message": "Critical: Large number of divergent transitions detected. Results likely invalid.",
            "divergent_transitions": diverging_count,
        }
    if diverging_percentage >= warning_threshold:
        return {
            "message": "Warning: Some divergent transitions detected. Results may be unreliable.",
            "divergent_transitions": diverging_count,
        }
    return {
        "message": "Divergent transitions are within acceptable limits.",
        "divergent_transitions": diverging_count,
    }
