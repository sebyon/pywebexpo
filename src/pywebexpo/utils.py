import numpy as np
from typing import Dict, Union
from model import WebExpoModel


def check_lognormal_data(data: np.ndarray, oel: float) -> bool:
    """
    Check if lognormal data is within the model limits.
    
    Conditions:
      1. All values are strictly positive.
      2. All values are between oel/1000 and 1000*oel.
    
    Args:
        data (np.ndarray): Data to check.
        oel (float): Occupational Exposure Limit.
        
    Returns:
        bool: True if data is within limits, False otherwise.
    """
    return np.all((data > 0) & (data >= oel / 1000) & (data <= 1000 * oel))


def check_normal_data(data: np.ndarray) -> bool:
    """
    Check if normal data is within the model limits.
    
    Conditions:
      1. All values are at least 40.
      2. All values are at most 140.
      3. All values are strictly positive.
    
    Args:
        data (np.ndarray): Data to check.
        
    Returns:
        bool: True if data is within limits, False otherwise.
    """
    # Note: data >= 40 implies data > 0, so the > 0 check is redundant.
    return np.all((data >= 40) & (data <= 140))


def inspect_divergent_traces(
    model: WebExpoModel, 
    warning_threshold: int = 1, 
    critical_threshold: int = 10
) -> Dict[str, Union[str, int]]:
    """
    Inspect divergent transitions after a WebExpoModel has been fit.
    
    Parameters
    ----------
    model : WebExpoModel
        The WebExpoModel object to inspect.
    warning_threshold : int, optional
        The percentage threshold (0 to 100) for issuing a warning if exceeded by divergent transitions.
    critical_threshold : int, optional
        The percentage threshold (0 to 100) for issuing a critical warning if exceeded by divergent transitions.
        
    Returns
    -------
    Dict[str, Union[str, int]]
        A dictionary containing a warning message and the number of divergent transitions.
    """
    if model.idata.sample_stats is None:
        raise ValueError("The model has not been fit yet. Please run the fit method first.")

    # Flatten the 'diverging' boolean array
    diverging = model.idata.sample_stats["diverging"].values.flatten()
    # Count the number of True values in diverging
    diverging_count = np.count_nonzero(diverging)
    total_transitions = diverging.size

    # Compute the percentage of divergent transitions
    diverging_percentage = (diverging_count / total_transitions) * 100

    if diverging_count == 0:
        return {"message": "No divergent transitions detected.", "divergent_transitions": 0}

    if diverging_percentage >= critical_threshold:
        return {
            "message": "Critical: Large number of divergent transitions detected. Results likely invalid.",
            "divergent_transitions": diverging_count
        }
    elif diverging_percentage >= warning_threshold:
        return {
            "message": "Warning: Some divergent transitions detected. Results may be unreliable.",
            "divergent_transitions": diverging_count
        }
    else:
        return {
            "message": "Divergent transitions are within acceptable limits.",
            "divergent_transitions": diverging_count
        }
