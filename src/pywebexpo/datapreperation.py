from dataclasses import dataclass
from typing import List, Optional, overload
import numpy as np
from logging import Logger

logger = Logger(__name__)

@dataclass
class WebExpoData:
    """
    Dataclass for the observed data in the WebExpo format. The data is stored as a list of values,
    with the corresponding lower and upper limits for each value. The data can be left-censored,
    right-censored, interval-censored, or uncensored.

    Additionally, the worker_id parameter can be used to specify the worker ID for each data point
    for the Between-Worker models.
    """
    y: List[float]
    worker_id: Optional[List[str]]
    lower_limits: List[float]
    upper_limits: List[float]

    def __post_init__(self):
        if len(self.y) != len(self.lower_limits) or len(self.y) != len(self.upper_limits):
            raise ValueError("The lengths of y, lower_limits, and upper_limits must be equal.")
            
        if self.worker_id is not None and len(self.y) != len(self.worker_id):
            raise ValueError("The lengths of y and worker_id must be equal.")
            
        if self.worker_id is None:
            self.worker_id = [None] * len(self.y)


# No worker ID / SEG models
@overload
def define_observed_lower_upper(expo_data: List[str], oel: float, worker_id: None = None) -> WebExpoData: ...

# Worker ID / Between Worker models
@overload
def define_observed_lower_upper(expo_data: List[str], oel: float, worker_id: List[str]) -> WebExpoData: ...


def define_observed_lower_upper(
    expo_data: List[str], 
    oel: float, 
    worker_id: Optional[List[str]] = None
) -> WebExpoData:
    """
    Define the observed data in the WebExpo format. The data is stored as a list of values,
    with the corresponding lower and upper limits for each value. The data can be left-censored,
    right-censored, interval-censored, or uncensored.

    The use of '<' and '>' symbols is used to denote left-censored and right-censored data,
    respectively. The '-' symbol is used to denote interval-censored data.

    Additionally, the worker_id parameter can be used to specify the worker ID for each data point
    for the Between-Worker models.

    Parameters
    ----------
    expo_data : List[str]
        The observed data values in the WebExpo format.
    oel : float
        The occupational exposure limit (OEL) for the data.
    worker_id : Optional[List[str]]
        The worker ID for each data point. If None, the worker ID is not used.

    Returns
    -------
    WebExpoData
        The observed data in the WebExpo format.
    """
    y = []
    lower_limits = []
    upper_limits = []

    # Process each value only once.
    for value in expo_data:
        if value.startswith('<'):
            # Left-censored: use the limit as the measurement and set the upper limit to infinity.
            limit = float(value[1:])
            y_val = limit
            lower_val = limit
            upper_val = np.inf
        elif value.startswith('>'):
            # Right-censored: use the limit as the measurement and set the lower limit to -infinity.
            limit = float(value[1:])
            y_val = limit
            lower_val = -np.inf
            upper_val = limit
        elif '-' in value and not value.startswith('-'):
            # Interval-censored: split into lower and upper limits.
            lower_val, upper_val = map(float, value.split('-', 1))
            y_val = (lower_val + upper_val) / 2
        else:
            # Uncensored: assign the value with infinite bounds.
            y_val = float(value)
            lower_val = -np.inf
            upper_val = np.inf

        y.append(y_val)
        lower_limits.append(lower_val)
        upper_limits.append(upper_val)

    # Pre-calculate the reciprocal of the OEL for scaling.
    inv_oel = 1 / oel

    # Scale the values by the OEL.
    # Dividing infinities by a finite positive number leaves them unchanged.
    y = [val * inv_oel for val in y]
    lower_limits = [val * inv_oel for val in lower_limits]
    upper_limits = [val * inv_oel for val in upper_limits]

    return WebExpoData(y, worker_id, lower_limits, upper_limits)
