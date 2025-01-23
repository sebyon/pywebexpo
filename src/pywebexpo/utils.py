import numpy as np


def check_lognormal_data(data: np.ndarray, oel:float) -> bool:
    """
    Check if lognormal data is within limits of model
    1. Between OEL/1000 abd 1000*OEL
    2. Strictly positive
    
    Args:
        data (np.ndarray): Data to check
        oel (float): Occupational Exposure Limit
        
    Returns:
        bool: True if data is within limits, False otherwise
    """

    if np.all(data > 0) and np.all(data >= oel/1000) and np.all(data <= 1000*oel):
        return True
    else:
        return False
    
def check_normal_data(data: np.ndarray) -> bool:
    """
    Check if normal data is within limits of model
    1. Between 40 and 140 for noise
    2. Strictly positive
    
    Args:
        data (np.ndarray): Data to check
        
    Returns:
        bool: True if data is within limits, False otherwise
    """
    # TODO: Check if easy to pass data type as argument for noise vs other data
    
    
    if np.all(data > 0) and np.all(data >= 40) and np.all(data <= 140):
        return True
    else:
        return False
    