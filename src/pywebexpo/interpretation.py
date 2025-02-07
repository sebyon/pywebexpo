import numpy as np
from dataclasses import dataclass
from typing import Dict, Union
from scipy.stats import norm
from functools import cached_property

@dataclass
class PointEstimates:
    """
    Calculate point estimates and risk metrics from posterior chains of log-transformed parameters.

    This class processes arrays of posterior samples for the log-transformed mean (mu_chain) and 
    log-transformed standard deviation (sigma_chain) to compute a variety of point estimates. 
    The estimates include:
      - Geometric mean and geometric standard deviation
      - Exceedance fraction relative to an occupational exposure limit (OEL)
      - A specified percentile of interest
      - Arithmetic mean
      - Fraction risk (percentage of samples exceeding a specified threshold)
      - Percentile risk (percentage of samples where the percentile estimate exceeds the OEL)
      - Arithmetic mean risk (percentage of samples where the arithmetic mean exceeds the OEL)
      - Arithmetic mean riskbands (distribution of arithmetic mean values across predefined risk bands)

    Parameters:
        mu_chain (np.ndarray): Array of posterior samples for the log-transformed mean.
        sigma_chain (np.ndarray): Array of posterior samples for the log-transformed standard deviation.
        probacred (int): Credible interval percentage (e.g., 95 for a 95% credible interval) used 
                         to derive the lower and upper credible limits.
        oel (float): Occupational exposure limit used as a threshold for various risk calculations.
        frac_threshold (float): Threshold (in percentage) used to determine the fraction risk.
        target_perc (float): Target percentile (e.g., 90 for the 90th percentile) for the percentile 
                             of interest calculation.
        precision (int, optional): Number of decimal places for rounding the calculated estimates. 
                                   Default is 3.
    """
    mu_chain: np.ndarray
    sigma_chain: np.ndarray
    probacred: int
    oel: float
    frac_threshold: float
    target_perc: float
    precision: int = 3

    def calculate_stats(self, chain: np.ndarray) -> Dict[str, float]:
        """
        Calculate the median, lower credible limit, and upper credible limit for the given chain.

        Parameters:
            chain (np.ndarray): The array of values to compute statistics on.

        Returns:
            Dict[str, float]: A dictionary with keys 'est', 'lcl', and 'ucl' representing
                              the median, lower credible limit, and upper credible limit respectively.
        """
        lower = 100 - self.probacred
        upper = self.probacred
        return {
            "est": round(np.median(chain), self.precision),
            "lcl": round(np.percentile(chain, lower), self.precision),
            "ucl": round(np.percentile(chain, upper), self.precision),
        }

    def _safe_sigma(self) -> np.ndarray:
        """
        Replace zeros in sigma_chain to avoid division by zero.

        Returns:
            np.ndarray: A version of sigma_chain where zeros have been replaced by a very small positive number.
        """
        return np.where(self.sigma_chain == 0, np.finfo(float).eps, self.sigma_chain)

    @cached_property
    def exp_mu(self) -> np.ndarray:
        """
        Compute and cache the exponentiated mu_chain.
        
        Returns:
            np.ndarray: The exponential of mu_chain.
        """
        return np.exp(self.mu_chain)

    @cached_property
    def exp_sigma(self) -> np.ndarray:
        """
        Compute and cache the exponentiated sigma_chain.
        
        Returns:
            np.ndarray: The exponential of sigma_chain.
        """
        return np.exp(self.sigma_chain)

    @cached_property
    def arithmetic_chain(self) -> np.ndarray:
        """
        Compute and cache the arithmetic chain.
        
        The arithmetic chain is defined as:
            arithmetic_chain = exp(mu_chain + 0.5 * sigma_chain**2)
        
        Returns:
            np.ndarray: The arithmetic chain values.
        """
        return np.exp(self.mu_chain + 0.5 * self.sigma_chain**2)

    @cached_property
    def percentile_chain(self) -> np.ndarray:
        """
        Compute and cache the percentile chain.
        
        The percentile chain is defined as:
            percentile_chain = exp(mu_chain + norm.ppf(target_perc / 100) * sigma_chain)
        
        Returns:
            np.ndarray: The percentile chain values.
        """
        return np.exp(self.mu_chain + norm.ppf(self.target_perc / 100) * self.sigma_chain)

    @cached_property
    def exceedance_chain(self) -> np.ndarray:
        """
        Compute and cache the exceedance chain.
        
        The exceedance chain is computed as:
            exceedance_chain = 100 * (1 - norm.cdf((log(oel) - mu_chain) / safe_sigma))
        where safe_sigma is sigma_chain with zeros replaced by a very small positive number.
        
        Returns:
            np.ndarray: The exceedance chain values.
        """
        safe_sigma = self._safe_sigma()
        return 100 * (1 - norm.cdf((np.log(self.oel) - self.mu_chain) / safe_sigma))

    def geometric_mean(self) -> Dict[str, float]:
        """
        Calculate the geometric mean statistics.
        
        Returns:
            Dict[str, float]: Statistics computed from the exponential of mu_chain.
        """
        return self.calculate_stats(self.exp_mu)

    def geometric_sd(self) -> Dict[str, float]:
        """
        Calculate the geometric standard deviation statistics.
        
        Returns:
            Dict[str, float]: Statistics computed from the exponential of sigma_chain.
        """
        return self.calculate_stats(self.exp_sigma)

    def exceedance_fraction(self) -> Dict[str, float]:
        """
        Calculate the exceedance fraction statistics.
        
        Returns:
            Dict[str, float]: Statistics computed from the exceedance chain.
        """
        return self.calculate_stats(self.exceedance_chain)

    def percentile_of_interest(self) -> Dict[str, float]:
        """
        Calculate the statistics for the percentile of interest.
        
        Returns:
            Dict[str, float]: Statistics computed from the percentile chain.
        """
        return self.calculate_stats(self.percentile_chain)

    def arithmetic_mean(self) -> Dict[str, float]:
        """
        Calculate the arithmetic mean statistics.
        
        Returns:
            Dict[str, float]: Statistics computed from the arithmetic chain.
        """
        return self.calculate_stats(self.arithmetic_chain)

    def frac_risk(self) -> float:
        """
        Calculate the fraction risk, which is the percentage of exceedance chain values 
        that exceed the specified fraction threshold.
        
        Returns:
            float: The fraction risk percentage.
        """
        return round(100 * np.mean(self.exceedance_chain > self.frac_threshold), self.precision)

    def perc_risk(self) -> float:
        """
        Calculate the percentile risk, which is the percentage of percentile chain values 
        that exceed the occupational exposure limit (OEL).
        
        Returns:
            float: The percentile risk percentage.
        """
        return round(100 * np.mean(self.percentile_chain > self.oel), self.precision)

    def am_risk(self) -> float:
        """
        Calculate the arithmetic mean risk, which is the percentage of arithmetic chain values 
        that exceed the occupational exposure limit (OEL).
        
        Returns:
            float: The arithmetic mean risk percentage.
        """
        return round(100 * np.mean(self.arithmetic_chain > self.oel), self.precision)

    def am_riskbands(self) -> Dict[str, float]:
        """
        Calculate the risk bands for the arithmetic mean.
        
        The arithmetic chain values are binned into predefined risk bands based on multiples of the OEL.
        
        Returns:
            Dict[str, float]: A dictionary where keys are the risk band labels and values are the corresponding percentages.
        """
        # Define bins spanning a very wide range using exponentials for flexibility.
        bins = [np.exp(-20), 0.01 * self.oel, 0.1 * self.oel, 0.5 * self.oel, self.oel, np.exp(20) * self.oel]
        categories = ["<0.01*OEL", "[0.01-0.1]*OEL", "[0.1-0.5]*OEL", "[0.5-1]*OEL", ">OEL"]
        counts, _ = np.histogram(self.arithmetic_chain, bins=bins)
        riskbands = 100 * counts / len(self.arithmetic_chain)
        return {cat: round(val, self.precision) for cat, val in zip(categories, riskbands)}

    def all_numeric(self) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Compile all point estimates and risk metrics into a single dictionary.
        
        Returns:
            Dict[str, Union[Dict[str, float], float]]: A dictionary containing all the computed statistics.
        """
        return {
            "gm": self.geometric_mean(),
            "gsd": self.geometric_sd(),
            "frac": self.exceedance_fraction(),
            "perc": self.percentile_of_interest(),
            "am": self.arithmetic_mean(),
            "frac.risk": self.frac_risk(),
            "perc.risk": self.perc_risk(),
            "am.risk": self.am_risk(),
            "am.riskbands": self.am_riskbands(),
        }
