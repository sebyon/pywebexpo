from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
import numpy as np
import pymc as pm
import arviz as az
import hashlib

class WebExpoModel(ABC):
    """
    Base class for the WebExpo models.
    
    Provides the basic structure for data preparation and model execution.
    """
    _model_type = "BaseWebExpoModel"
    _dist_type = "None"
    version = "None"
    
    # Use __slots__ to reduce per-instance overhead
    __slots__ = ("model", "idata", "oel")
    
    def __init__(self, **kwargs):
        self.model = None
        self.idata: Optional[az.InferenceData] = None
    
    def set_oel(self, oel: float) -> None:
        """Set the occupational exposure limit (OEL) for the model."""
        self.oel = oel
    
    @property
    @abstractmethod
    def get_default_error_mode(self) -> str:
        """
        Return the default error mode for the model.
        
        For example, "CV", "SD", or None.
        """
        ...
    
    @staticmethod
    @abstractmethod
    def get_default_model_config() -> Dict:
        """
        Return the default model configuration for the model.
        """
        ...
    
    @staticmethod
    @abstractmethod
    def get_default_sampler_config() -> Dict:
        """
        Return the default sampler configuration for the model.
        """
        ...
    
    @staticmethod
    @abstractmethod
    def get_default_analysis_config() -> Dict:
        """
        Return the default analysis configuration for the model.
        """
        ...
    
    @abstractmethod
    def _generate_and_preprocess_model_data(
        self, data: np.ndarray, oel: float
    ) -> Any:
        """
        Preprocess the raw data and prepare it for the model.
        """
        ...
    
    @abstractmethod
    def build_model(self, **kwargs) -> None:
        """
        Build the PyMC model.
        """
        ...
    
    @abstractmethod 
    def sample_model(self, **kwargs) -> None:
        """
        Sample from the PyMC model.
        """
        ...
    
    @abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Fit the model using the PyMC sampler.
        """
        ...
    
    def set_idata_attrs(self, idata: Optional[az.InferenceData] = None) -> az.InferenceData:
        """
        Set attributes on an InferenceData object.
        """
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No InferenceData provided to set attributes on.")
        idata.attrs.update({
            "id": self.id,
            "model_type": self._model_type,
            "version": self.version,
        })
        if hasattr(self, "_save_input_params"):
            self._save_input_params(idata)
        return idata
    
    @property
    def id(self) -> str:
        """
        Return a unique model id based on the version and model type.
        """
        hasher = hashlib.md5()
        hasher.update(str(self.version).encode())
        hasher.update(str(self._model_type).encode())
        return hasher.hexdigest()[:16]
    
    @abstractmethod
    def _extract_chains(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chains for the key parameters (e.g., mu and sigma) from the model trace.
        """
        ...
    
    @abstractmethod
    def analyse_chains(self) -> Dict:
        """
        Analyze the chains from the model.
        """
        ...
