from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import arviz as az
import hashlib


class WebExpoModel(ABC):
    """Base class for WebExpo models providing the build → sample → analyse lifecycle."""

    _model_type = "BaseWebExpoModel"
    _dist_type = "None"
    version = "None"

    def __init__(self, **kwargs):
        self.model = None
        self.idata: az.InferenceData | None = None

    def set_oel(self, oel: float) -> None:
        """Set the occupational exposure limit (OEL)."""
        self.oel = oel

    @staticmethod
    @abstractmethod
    def get_default_error_mode() -> str | None:
        """Return the default error mode (e.g. 'CV', 'SD', or None)."""
        ...

    @staticmethod
    @abstractmethod
    def get_default_model_config() -> dict:
        """Return the default model configuration."""
        ...

    @staticmethod
    @abstractmethod
    def get_default_sampler_config() -> dict:
        """Return the default sampler configuration."""
        ...

    @staticmethod
    @abstractmethod
    def get_default_analysis_config() -> dict:
        """Return the default analysis configuration."""
        ...

    @abstractmethod
    def _generate_and_preprocess_model_data(self) -> None:
        """Preprocess raw data and populate model data attributes."""
        ...

    @abstractmethod
    def build_model(self, **kwargs) -> None:
        """Build the PyMC model."""
        ...

    @abstractmethod
    def sample_model(self, **kwargs) -> None:
        """Sample from the PyMC model."""
        ...

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """Build and sample the model."""
        ...

    def set_idata_attrs(self, idata: az.InferenceData | None = None) -> az.InferenceData:
        """Set model metadata attributes on an InferenceData object."""
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
        """Unique model ID derived from version and model type."""
        hasher = hashlib.md5()
        hasher.update(str(self.version).encode())
        hasher.update(str(self._model_type).encode())
        return hasher.hexdigest()[:16]

    @abstractmethod
    def _extract_chains(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract key parameter chains (mu and sigma) from the posterior."""
        ...

    @abstractmethod
    def analyse_chains(self) -> dict:
        """Analyse the posterior chains and return a results dictionary."""
        ...
