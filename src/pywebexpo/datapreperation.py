import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class WebExpoData:
    #TODO: Figure out if use one dataclass for all models or one for SEG/Worker
    
    y: List[float]
    lower_limits: List[float]
    upper_limits: List[float]

def define_observed_lower_upper(expo_data: List[str], oel:float) -> WebExpoData:
    y = []
    lower_limits = []
    upper_limits = []

    for value in expo_data:
        if value.startswith('<'):
            # Left-censored
            limit = float(value[1:])  # Extract the numeric value
            y.append(limit)
            lower_limits.append(limit)
            upper_limits.append(np.inf)
        elif value.startswith('>'):
            # Right-censored
            limit = float(value[1:])  # Extract the numeric value
            y.append(limit)
            lower_limits.append(-np.inf)
            upper_limits.append(limit)
        elif '-' in value:
            # Interval-censored
            lower, upper = map(float, value.split('-'))  # Extract the range
            y.append((lower + upper) / 2)
            lower_limits.append(lower)
            upper_limits.append(upper)
        else:
            # Uncensored
            y.append(float(value))
            lower_limits.append(-np.inf)
            upper_limits.append(np.inf)
        
    # Divide all values by the OEL, excluding infinite values
    y = [value / oel if value != np.inf and value != -np.inf else value for value in y]
    lower_limits = [value / oel if value != np.inf and value != -np.inf else value for value in lower_limits]
    upper_limits = [value / oel if value != np.inf and value != -np.inf else value for value in upper_limits]

    
    return WebExpoData(y, lower_limits, upper_limits)
    
    
    
    
