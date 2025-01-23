import numpy as np
from dataclasses import dataclass
from typing import Dict, Union
from scipy.stats import norm
import arviz as az
from typing import Tuple

#TODO: Clean up the type hints

@dataclass
class PointEstimates:
    mu_chain: np.ndarray
    sigma_chain: np.ndarray
    probacred: int
    oel: float
    frac_threshold: float
    target_perc: float
    precision: int = 3

    @staticmethod
    def calculate_stats(
        chain: np.ndarray, 
        probacred: int, 
        precision: int = 3) -> Dict[str, float]:
        lower = 100 - probacred
        upper = probacred
        return {
            "est": round(np.median(chain), precision),
            "lcl": round(np.percentile(chain, lower), precision),
            "ucl": round(np.percentile(chain, upper), precision),
        }
    

    def geometric_mean(self) -> Dict[str, float]:
        return self.calculate_stats(np.exp(self.mu_chain), self.probacred, self.precision)

    def geometric_sd(self) -> Dict[str, float]:
        return self.calculate_stats(np.exp(self.sigma_chain), self.probacred, self.precision)

    def exceedance_fraction(self) -> Dict[str, float]:
        chain = 100 * (1 - norm.cdf((np.log(self.oel) - self.mu_chain) / self.sigma_chain))
        return self.calculate_stats(chain, self.probacred, self.precision)

    def percentile_of_interest(self) -> Dict[str, float]:
        chain = np.exp(self.mu_chain + norm.ppf(self.target_perc / 100) * self.sigma_chain)
        return self.calculate_stats(chain, self.probacred, self.precision)

    def arithmetic_mean(self) -> Dict[str, float]:
        chain = np.exp(self.mu_chain + 0.5 * self.sigma_chain**2)
        return self.calculate_stats(chain, self.probacred, self.precision)

    def frac_risk(self) -> float:
        chain = 100 * (1 - norm.cdf((np.log(self.oel) - self.mu_chain) / self.sigma_chain))
        return round(100 * np.sum(chain > self.frac_threshold) / len(chain), self.precision)

    def perc_risk(self) -> float:
        chain = np.exp(self.mu_chain + norm.ppf(self.target_perc / 100) * self.sigma_chain)
        return round(100 * np.sum(chain > self.oel) / len(chain), self.precision)

    def am_risk(self) -> float:
        chain = np.exp(self.mu_chain + 0.5 * self.sigma_chain**2)
        return round(100 * np.sum(chain > self.oel) / len(chain), self.precision)

    def am_riskbands(self) -> Dict[str, float]:
        chain = np.exp(self.mu_chain + 0.5 * self.sigma_chain**2)
        riskbands = 100 * np.histogram(
            chain, 
            bins=[
                np.exp(-20), 
                0.01 * self.oel, 
                0.1 * self.oel, 
                0.5 * self.oel, 
                self.oel, 
                np.exp(20) * self.oel
                ]
            )[0] / len(chain)
        categories = ["<0.01*OEL", "[0.01-0.1]*OEL", "[0.1-0.5]*OEL", "[0.5-1]*OEL", ">OEL"]
        return {category: round(risk, self.precision) for category, risk in zip(categories, riskbands)}

    def all_numeric(self) -> Dict[str, Union[Dict[str, float], float]]:
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
