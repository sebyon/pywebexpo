import numpy as np
import pymc as pm
from typing import List, Dict, Any, Tuple
from model import WebExpoModel
from datapreperation import define_observed_lower_upper
from interpretation import PointEstimates

__all__ = ["SEGInformedVarLognormal"]

class SEGInformedVarLognormal(WebExpoModel):
    """_summary_

    Args:
        WebExpoModel (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    _model_type = "InformedVar"
    _dist_type = "Lognormal"
    version = "0.1"
    
    
    def __init__(
        self,
        data: List[str],
        oel: float = None,
        sample_config: Dict | None = None,
        model_config: Dict | None = None,
        error_mode: str | None = None,
        analysis_config: Dict | None = None
    ):
        
        
        super().__init__()
        
        sampler_config = self.get_default_sample_config() if sample_config is None else sample_config
        self.sampler_config = sampler_config
        
        model_config = self.get_default_model_config() if model_config is None else model_config
        self.model_config = model_config
        
        error_mode = self.get_default_error_mode() if error_mode is None else error_mode
        self.error_mode = error_mode
        
        analysis_config = self.get_default_analysis_config() if analysis_config is None else analysis_config
        self.analysis_config = analysis_config
        
        self.data = data
        self.oel = oel
        
    
    def build_model(self, **kwargs) -> pm.Model:
        r"""
        Builds the model for the InformedVar model.

        Args:
            data (Dict[str, Any]): A dictionary containing the data for the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            pm.Model: The PyMC model for the InformedVar model.
        """
        
        self._generate_and_preprocess_model_data()

        with pm.Model() as self.model:
            # Shared components
            y_data = pm.Data("y_data", self.y_data)
            
            mu = pm.Uniform("mu", lower=self.model_config["mu_lower"], upper=self.model_config["mu_upper"])
            log_sigma = pm.Normal("log_sigma", mu=self.model_config["log_sigma_mu"], sigma= 1/pm.math.sqrt(self.model_config["log_sigma_prec"]))
            sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
            
            # Determine measurement error based on error_mode
            
            
            
            if self.error_mode == "CV":
            
                cv = pm.Uniform("cv", lower=self.model_config["error_lower"], upper=self.model_config["error_upper"])
                true_data = pm.Lognormal("true_data", mu=mu, sigma=sigma, shape=y_data.shape)
                meas_error = pm.Deterministic("meas_error", (1 / (cv * true_data)**2))
                
                censored_observations = pm.Censored(
                    "y with censoring",
                    pm.Normal.dist(mu=true_data, tau=meas_error, shape=y_data.shape),
                    lower=self.lower_limits,
                    upper=self.upper_limits,
                    observed=y_data,
                )
                    
            elif self.error_mode == "SD":
            
                sd = pm.Uniform("sd", lower=self.model_config["error_lower"], upper=self.model_config["error_upper"])
                true_data = pm.Lognormal("true_data", mu=mu, sigma=sigma, shape=y_data.shape)
                
                censored_observations = pm.Censored(
                    "y with censoring",
                    pm.Normal.dist(mu=true_data, sigma=sd, shape=y_data.shape),
                    lower=self.lower_limits,
                    upper=self.upper_limits,
                    observed=y_data,
                )

            elif self.error_mode is None:
                # Default case uses LogNormal distribution
                observed_dist = pm.Lognormal.dist(mu=mu, sigma=sigma)
                
                censored_observations = pm.Censored(
                    "y",
                    observed_dist,
                    lower=self.lower_limits,
                    upper=self.upper_limits,
                    observed=y_data,
                )
                
            else :
                raise ValueError(f"Invalid error mode: {self.error_mode}")

            # Define censored observed data

        return self.model
    
    def fit(self, **kwargs) -> pm.Model:
        r"""
        Fits the InformedVar model to the provided data.

        Args:
            data (Dict[str, Any]): A dictionary containing the data for the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            pm.Model: The PyMC model for the InformedVar model.
        """
        
        return super().fit(**kwargs)
        
    def set_oel(self, oel):
        return super().set_oel(oel)
    
    def sample_model(self, **kwargs):
        if self.model is None:
            raise RuntimeError("Model is not built. Please build the model before sampling.")
        
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            print(sampler_args)
            idata = pm.sample(**sampler_args)
            self.idata = idata
        return self
    
    
    @staticmethod
    def get_default_error_mode() -> str | None:
        r"""
        Returns the default error mode for the InformedVar model.
        
        Returns:
            str: The default error mode for the InformedVar model.
        """
        return None
    
    @staticmethod
    def get_default_model_config() -> Dict:
        r"""
        Returns the default model configuration for the InformedVar model.
        
        Returns:
            Dict: The default model configuration for the InformedVar model.
        """
        model_config = {
            "mu_lower": -20.0,
            "mu_upper": 20,
            "log_sigma_mu": -0.1744,
            "log_sigma_prec": 2.5523,
            "error_lower": 0.30,
            "error_upper": 0.300001
        }
        return model_config
    
    
    @staticmethod
    def get_default_sample_config() -> Dict:
        r"""
        Returns the default sample configuration for the InformedVar model.
        
        Returns:
            Dict: The default sample configuration for the InformedVar model.
        """
        sample_config = {
            "draws": 4000,
            "tune": 4000,
            "chains": 6,
            "target_accept": 0.85,
            #"return_inferencedata": True,
            "nuts_sampler": "numpyro"
        }
        return sample_config
    
    @staticmethod
    def get_default_analysis_config() -> Dict:
        r"""
        Returns the default analysis configuration for the InformedVar model.
        
        Returns:
            Dict: The default analysis configuration for the InformedVar model.
        """
        
        analysis_config = {
            "probacred": 90,
            "frac_threshold": 10,
            "target_perc": 95
        }
        
        return analysis_config
    
    def _generate_and_preprocess_model_data(self) -> None:
        """
        Generates and preprocesses the model data for the InformedVar model based on the provided y_values.
        The data is divided into three categories: censored, uncensored, and limits for use in the model.
        
        Example:
        >>>     y_values = ["<5", "10-20", ">30", "25"]
        >>>     
        >>>
        
        Args:
            y_values (List[str]): _description_

        Returns:
            _type_: _description_
        """

        
        data = define_observed_lower_upper(self.data, self.oel)
        
        self.y_data = np.array(data.y)
        self.lower_limits = np.array(data.lower_limits)
        self.upper_limits = np.array(data.upper_limits)
        
        return None
    
    @staticmethod
    def _extract_chains(self) -> Tuple[np.ndarray, np.ndarray]:
        
        mu_chain = self.idata.posterior["mu"].values.flatten()
        sigma_chain = self.idata.posterior["sigma"].values.flatten()
        
        # Add log(oel) to each element in the mu chain
        mu_chain += np.log(self.oel)
        return mu_chain, sigma_chain
    
    def analyse_chains(self) -> PointEstimates:
        mu_chain, sigma_chain = self._extract_chains(self)
        
        analysis = PointEstimates(mu_chain, 
                                       sigma_chain,
                                       self.analysis_config["probacred"],
                                       self.oel,
                                       self.analysis_config["frac_threshold"],
                                       self.analysis_config["target_perc"])
        
        self.analysis = analysis.all_numeric()
        
        return self.analysis
        
    
    
        
        
    
    
        
        
    
    