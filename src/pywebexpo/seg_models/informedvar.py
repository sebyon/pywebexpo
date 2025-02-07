import numpy as np
import pymc as pm
from typing import List, Dict, Tuple, Optional
from model import WebExpoModel
from datapreperation import define_observed_lower_upper
from interpretation import PointEstimates
import logging

__all__ = ["SEGInformedVarLognormal"]

logger = logging.getLogger(__name__)

class SEGInformedVarLognormal(WebExpoModel):
    """
    InformedVar model in the SEG framework using a Lognormal distribution.
    
    Example:
        model = SEGInformedVarLognormal(data=error_data, oel=100, error_mode='CV')
        model.build_model()
        model.sample_model()
        model.analyse_chains()
    
    Args:
        data (List[str]): Data in WebExpo format.
        oel (float): Occupational exposure limit.
        sample_config (Optional[Dict]): Sampler configuration.
        model_config (Optional[Dict]): Model configuration.
        error_mode (Optional[str]): Error mode.
        analysis_config (Optional[Dict]): Analysis configuration.
    """
    
    _model_type = "InformedVar"
    _dist_type = "Lognormal"
    version = "0.1"
    
    def __init__(
        self,
        data: List[str],
        oel: float,
        sample_config: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        error_mode: Optional[str] = None,
        analysis_config: Optional[Dict] = None
    ):
        super().__init__()
        self.sampler_config = sample_config or self.get_default_sampler_config()
        self.model_config = model_config or self.get_default_model_config()
        self.error_mode = error_mode if error_mode is not None else self.get_default_error_mode()
        self.analysis_config = analysis_config or self.get_default_analysis_config()
        self.data = data
        self.oel = oel
    
    def build_model(self) -> pm.Model:
        """
        Build the InformedVar Lognormal model.
        
        Returns:
            pm.Model: The built PyMC model.
        """
        if not self.data:
            raise ValueError("Data is not provided. Provide data before building the model.")
        if self.oel is None:
            raise ValueError("OEL is not provided. Provide an OEL before building the model.")
        
        self._generate_and_preprocess_model_data()
        
        with pm.Model() as model:
            self.model = model  # Save the model for later use.
            # Shared data
            y_data = pm.Data("y_data", self.y_data)
            
            # Priors
            mu = pm.Uniform("mu", lower=self.model_config["mu_lower"], upper=self.model_config["mu_upper"])
            log_sigma = pm.Normal("log_sigma", mu=self.model_config["log_sigma_mu"],
                                  sigma=1/pm.math.sqrt(self.model_config["log_sigma_prec"]))
            sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
            
            # Likelihood based on error_mode
            if self.error_mode == "CV":
                cv = pm.Uniform("cv", lower=self.model_config["error_lower"], upper=self.model_config["error_upper"])
                true_data = pm.Lognormal("true_data", mu=mu, sigma=sigma, shape=y_data.shape)
                meas_error = pm.Deterministic("meas_error", 1 / (cv * true_data)**2)
                likelihood = pm.Normal.dist(mu=true_data, tau=meas_error, shape=y_data.shape)
            elif self.error_mode == "SD":
                sd = pm.Uniform("sd", lower=self.model_config["error_lower"], upper=self.model_config["error_upper"])
                true_data = pm.Lognormal("true_data", mu=mu, sigma=sigma, shape=y_data.shape)
                likelihood = pm.Normal.dist(mu=true_data, sigma=sd, shape=y_data.shape)
            elif self.error_mode is None:
                likelihood = pm.Lognormal.dist(mu=mu, sigma=sigma)
            else:
                raise ValueError(f"Invalid error mode: {self.error_mode}")
            
            pm.Censored(
                "observed_data",
                likelihood,
                lower=self.lower_limits,
                upper=self.upper_limits,
                observed=y_data
            )
        return self.model
    
    def sample_model(self, **kwargs) -> "SEGInformedVarLognormal":
        """
        Sample the model using the configured sampler.
        
        Returns:
            SEGInformedVarLognormal: self, with the `idata` attribute updated.
        """
        if self.model is None:
            raise RuntimeError("Model is not built. Build the model before sampling.")
        
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            logger.debug("Sampler arguments: %s", sampler_args)
            self.idata = pm.sample(**sampler_args)
        return self
    
    def fit(self) -> pm.Model:
        """
        Fit the model by building and sampling.
        
        Returns:
            pm.Model: The fitted model.
        """
        self.build_model()
        self.sample_model()
        return self.model
    
    @staticmethod
    def get_default_error_mode() -> Optional[str]:
        return None
    
    @staticmethod
    def get_default_model_config() -> Dict:
        return {
            "mu_lower": -20.0,
            "mu_upper": 20.0,
            "log_sigma_mu": -0.1744,
            "log_sigma_prec": 2.5523,
            "error_lower": 0.15,
            "error_upper": 0.45
        }
    
    @staticmethod
    def get_default_sampler_config() -> Dict:
        return {
            "draws": 5000,
            "tune": 4000,
            "chains": 6,
            "target_accept": 0.85,
            "nuts_sampler": "numpyro",
            "initvals": {
                "mu": np.log(0.3), 
                "log_sigma": np.log(2.5)
            },
        }
    
    @staticmethod
    def get_default_analysis_config() -> Dict:
        return {
            "probacred": 90,
            "frac_threshold": 10,
            "target_perc": 95
        }
    
    def _generate_and_preprocess_model_data(self) -> None:
        """
        Generate and preprocess model data.
        Sets the attributes:
            - y_data: the observed data as a numpy array.
            - lower_limits, upper_limits: censoring bounds as numpy arrays.
        """
        data = define_observed_lower_upper(expo_data=self.data, oel=self.oel)
        self.y_data = np.array(data.y)
        self.lower_limits = np.array(data.lower_limits)
        self.upper_limits = np.array(data.upper_limits)
    
    def _extract_chains(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the mu and sigma chains from the model trace.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The mu_chain and sigma_chain.
        """
        mu_chain = self.idata.posterior["mu"].values.flatten()
        sigma_chain = self.idata.posterior["sigma"].values.flatten()
        # Adjust the mu chain with log(oel) as required by the model formulation.
        mu_chain += np.log(self.oel)
        return mu_chain, sigma_chain
    
    def analyse_chains(self) -> Dict:
        """
        Analyze the chains using the PointEstimates class.
        
        Returns:
            Dict: A dictionary of chain analysis results.
        """
        mu_chain, sigma_chain = self._extract_chains()
        analysis = PointEstimates(
            mu_chain,
            sigma_chain,
            self.analysis_config["probacred"],
            self.oel,
            self.analysis_config["frac_threshold"],
            self.analysis_config["target_perc"]
        )
        self.analysis = analysis.all_numeric()
        return self.analysis
