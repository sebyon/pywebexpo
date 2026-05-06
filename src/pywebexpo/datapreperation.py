from dataclasses import dataclass
from typing import overload
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WebExpoData:
    """
    Observed data in WebExpo format with per-value censoring bounds.

    Supports left-censored ('<'), right-censored ('>'), interval-censored ('lower-upper'),
    and uncensored values. worker_id is only used by Between-Worker models.
    """

    y: list[float]
    worker_id: list[str | None] | None
    lower_limits: list[float]
    upper_limits: list[float]

    def __post_init__(self):
        if len(self.y) != len(self.lower_limits) or len(self.y) != len(self.upper_limits):
            raise ValueError("The lengths of y, lower_limits, and upper_limits must be equal.")

        if self.worker_id is not None and len(self.y) != len(self.worker_id):
            raise ValueError("The lengths of y and worker_id must be equal.")

        if self.worker_id is None:
            self.worker_id = [None] * len(self.y)


# No worker ID / SEG models
@overload
def define_observed_lower_upper(expo_data: list[str], oel: float, worker_id: None = None) -> WebExpoData: ...

# Worker ID / Between-Worker models
@overload
def define_observed_lower_upper(expo_data: list[str], oel: float, worker_id: list[str]) -> WebExpoData: ...


def define_observed_lower_upper(
    expo_data: list[str],
    oel: float,
    worker_id: list[str] | None = None,
) -> WebExpoData:
    """
    Parse WebExpo-format strings into observed values and censoring bounds, scaled by OEL.

    Censoring notation:
    - '<value'    — left-censored
    - '>value'    — right-censored
    - 'low-high'  — interval-censored
    - numeric     — uncensored

    Parameters
    ----------
    expo_data : list[str]
        Observed data values in WebExpo format.
    oel : float
        Occupational exposure limit used to scale values.
    worker_id : list[str] | None
        Worker IDs for Between-Worker models. None for SEG models.

    Returns
    -------
    WebExpoData
        Parsed and OEL-scaled data with censoring bounds.
    """
    y = []
    lower_limits = []
    upper_limits = []

    for value in expo_data:
        if value.startswith('<'):
            limit = float(value[1:])
            y_val, lower_val, upper_val = limit, limit, np.inf
        elif value.startswith('>'):
            limit = float(value[1:])
            y_val, lower_val, upper_val = limit, -np.inf, limit
        elif '-' in value and not value.startswith('-'):
            lower_val, upper_val = map(float, value.split('-', 1))
            y_val = (lower_val + upper_val) / 2
        else:
            y_val = float(value)
            lower_val, upper_val = -np.inf, np.inf

        y.append(y_val)
        lower_limits.append(lower_val)
        upper_limits.append(upper_val)

    inv_oel = 1 / oel
    y = [val * inv_oel for val in y]
    lower_limits = [val * inv_oel for val in lower_limits]
    upper_limits = [val * inv_oel for val in upper_limits]

    return WebExpoData(y, worker_id, lower_limits, upper_limits)
