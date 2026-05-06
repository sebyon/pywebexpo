import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from functools import cached_property


@dataclass
class PointEstimates:
    """
    Point estimates and risk metrics derived from posterior MCMC chains.

    Parameters:
        mu_chain: Posterior samples for the log-transformed mean.
        sigma_chain: Posterior samples for the log-transformed standard deviation.
        probacred: Credible interval percentage (e.g. 90 for a 90% CI).
        oel: Occupational exposure limit.
        frac_threshold: Exceedance fraction threshold (%).
        target_perc: Target percentile (e.g. 95 for the 95th percentile).
        precision: Decimal places for rounding. Default 3.
    """

    mu_chain: np.ndarray
    sigma_chain: np.ndarray
    probacred: int
    oel: float
    frac_threshold: float
    target_perc: float
    precision: int = 3

    def calculate_stats(self, chain: np.ndarray) -> dict[str, float]:
        """Return median, lower credible limit, and upper credible limit for a chain."""
        lower = 100 - self.probacred
        upper = self.probacred
        return {
            "est": round(np.median(chain), self.precision),
            "lcl": round(np.percentile(chain, lower), self.precision),
            "ucl": round(np.percentile(chain, upper), self.precision),
        }

    def _safe_sigma(self) -> np.ndarray:
        """Return sigma_chain with zeros replaced by eps to avoid division by zero."""
        return np.where(self.sigma_chain == 0, np.finfo(float).eps, self.sigma_chain)

    @cached_property
    def exp_mu(self) -> np.ndarray:
        return np.exp(self.mu_chain)

    @cached_property
    def exp_sigma(self) -> np.ndarray:
        return np.exp(self.sigma_chain)

    @cached_property
    def arithmetic_chain(self) -> np.ndarray:
        return np.exp(self.mu_chain + 0.5 * self.sigma_chain**2)

    @cached_property
    def percentile_chain(self) -> np.ndarray:
        return np.exp(self.mu_chain + norm.ppf(self.target_perc / 100) * self.sigma_chain)

    @cached_property
    def exceedance_chain(self) -> np.ndarray:
        safe_sigma = self._safe_sigma()
        return 100 * (1 - norm.cdf((np.log(self.oel) - self.mu_chain) / safe_sigma))

    def geometric_mean(self) -> dict[str, float]:
        return self.calculate_stats(self.exp_mu)

    def geometric_sd(self) -> dict[str, float]:
        return self.calculate_stats(self.exp_sigma)

    def exceedance_fraction(self) -> dict[str, float]:
        return self.calculate_stats(self.exceedance_chain)

    def percentile_of_interest(self) -> dict[str, float]:
        return self.calculate_stats(self.percentile_chain)

    def arithmetic_mean(self) -> dict[str, float]:
        return self.calculate_stats(self.arithmetic_chain)

    def frac_risk(self) -> float:
        return round(100 * np.mean(self.exceedance_chain > self.frac_threshold), self.precision)

    def perc_risk(self) -> float:
        return round(100 * np.mean(self.percentile_chain > self.oel), self.precision)

    def am_risk(self) -> float:
        return round(100 * np.mean(self.arithmetic_chain > self.oel), self.precision)

    def am_riskbands(self) -> dict[str, float]:
        """Bin the arithmetic chain into predefined OEL-relative risk bands."""
        bins = [np.exp(-20), 0.01 * self.oel, 0.1 * self.oel, 0.5 * self.oel, self.oel, np.exp(20) * self.oel]
        categories = ["<0.01*OEL", "[0.01-0.1]*OEL", "[0.1-0.5]*OEL", "[0.5-1]*OEL", ">OEL"]
        counts, _ = np.histogram(self.arithmetic_chain, bins=bins)
        riskbands = 100 * counts / len(self.arithmetic_chain)
        return {cat: round(val, self.precision) for cat, val in zip(categories, riskbands)}

    def all_numeric(self) -> dict[str, dict[str, float] | float]:
        """Compile all point estimates and risk metrics into a single dictionary."""
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
