#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#  This file is a modified version of the original pymc_extras.model_builder module
#  from the PyMC project. It has been modified to fit the needs of the WebExpo project

# Modified by: Sebastian D'Hyon
# Date: 2025-01-04
# Description: This module contains the abstract class for the WebExpo models based 

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datapreperation import WebExpoData
from interpretation import PointEstimates
import numpy as np
import pymc as pm
import arviz as az
import hashlib
import json


class WebExpoModel():

    """
    Base class for the WebExpo models.
    
    Provides the basic structure for the models, including the data preparation and model execution.
    
    """
    
    _model_type = "BaseWebExpoModel"
    _dist_type = "None"
    version = "None"
    
    def __init__(self, **kwargs):
        """
        Initializes model configuration and sampler configuration for the model

        Parameters
        ----------
        data : Dictionary, optional
            It is the data we need to train the model on.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
        error_mode : String, optional
            string containing the error mode to use. The avaliable options are "CV", "SD" or None

        Examples
        --------
        >>> model_config = {"mu_lower": -20.0, "mu_upper": 20, "log_sigma_mu": 0, "log_sigma_sigma": 1, "error_lower": 0.15, "error_upper": 0.45}
        >>> sampler_config = {"draws": 25_000, "tune": 5_000, "chains": 4, "target_accept": 0.90, "nuts_sampler": "blackjax"}
        >>> error_mode = "CV"
        >>> model = MyModel(model_config, sampler_config, error_mode)
        """

        self.model = None
        self.idata: az.InferenceData | None = None
        

    def set_oel(self, oel: float) -> None:
        """
        Sets the occupational exposure limit (OEL) for the model.
        
        Parameters
        ----------
        oel: float
            The occupational exposure limit (OEL) for the model.
        """
        self.oel = oel
        
    
    @property
    @abstractmethod
    def get_default_error_mode(self) -> str:
        """
        Returns the error mode of the model. The error mode is used to determine the error calculation method.
        
        The available options are:
        - "CV": Coefficient of Variation
        - "SD": Standard Deviation
        - None: No error calculation
        
        Returns:
        ---------
        error_mode: str
            The error mode of the model.
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def get_default_model_config() -> Dict:
        """
        Returns the default model configuration for the WebExpo Model.
        This configuration is used to create the model parameters if no configuration is provided.
        
        Example for the InformedVar model:
        
        .. code-block:: python
            @staticmethod
            def get_default_model_config() -> Dict:
                return {
                    "mu_lower": -20.0,
                    "mu_upper": 20,
                    "log_sigma_mu": 0,
                    "log_sigma_sigma": 1
                    "error_lower": 0.15,
                    "error_upper": 0.45
                }
            
        Returns:
        ---------
        model_config: Dict
            The default model configuration for the WebExpo Model.
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def get_default_sampler_config(self) -> dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization
        Useful for understanding structure of required sampler_config to allow its customization by users

        Examples
        --------
        >>>     @staticmethod
        >>>     def default_sampler_config():
        >>>         return {
        >>>             'draws': 25_000,
        >>>             'tune': 5_000,
        >>>             'chains': 4,
        >>>             'target_accept': 0.90,
        >>>             'nuts_sampler': 'blackjax',
        >>>         }

        Returns
        -------
        sampler_config : dict
            A set of default settings for used by model in fit process.
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def get_default_analysis_config(self) -> dict:
        """
        Returns a class default analysis dict for model builder if no analysis_config is provided on class initialization
        Useful for understanding structure of required analysis_config to allow its customization by users

        Examples
        --------
        >>>     @staticmethod
        >>>     def default_analysis_config():
        >>>         return {
        >>>             'probacred': 0.95,
        >>>             'frac_threshold': 1,
        >>>             'target_perc': 95,
        >>>         }

        Returns
        -------
        analysis_config : dict
            A set of default settings for used by model in fit process.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_and_preprocess_model_data(
        self,
        data: np.ndarray,
        oel: float,
    ) -> WebExpoData:
        """
        Applies preprocessing to the data before fitting the model.
        if validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        In case of optional parameters being passed into the model, this method should implement the conditional
        logic responsible for correct handling of the optional parameters, and including them into the dataset.

        Parameters
        ----------
        X : array, shape (n_obs, n_features)
        y : array, shape (n_obs,)

        Examples
        --------
        >>>     @classmethod
        >>>     def _generate_and_preprocess_model_data(self, X, y):
                    coords = {
                        'x_dim': X.dim_variable,
                    } #only include if applicable for your model
        >>>         self.X = X
        >>>         self.y = y

        Returns
        -------
        None

        """
        raise NotImplementedError
    
    @abstractmethod
    def build_model(
        self,
        **kwargs,
    ) -> None:
        """
        Creates an instance of pm.Model based on provided data and model_config, and
        attaches it to self.

        Parameters
        ----------

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.

        See Also
        --------
        default_model_config : returns default model config

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented in a subclass.
        """
        raise NotImplementedError
    
    @abstractmethod 
    def sample_model(self, **kwargs) -> None:
        """
        Samples the model using the PyMC3 sampler.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to the PyMC3 sampler.
            
        """
        raise NotImplementedError
    
    def fit(
        self,
        progressbar: bool = True,
        random_seed: int | None = None,
        **kwargs: Any,
        ) -> az.InferenceData:
        """
        Fits the model to the provided data.

        Args:
            data (CensoredData): The censored data to fit the model to.
            progressbar (bool): Whether to display a progress bar during sampling.
            predictor_names (List[str]): The names of the predictors to include in the model.
            random_seed (RandomState | int): The random seed to use for sampling.
            **kwargs: Additional keyword arguments to pass to the sampling function.

        Returns:
            az.InferenceData: The ArviZ InferenceData object containing the model trace.
        """
        if self.oel is None:
            raise ValueError("OEL is not set. Please set the OEL before fitting the model by using Model.set_oel().")
        
        self._generate_and_preprocess_model_data(data.y, self.oel)
        self.model = self.build_model(self.y_data)
        
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)
    
    def set_idata_attrs(self, idata=None):
        """
        Set attributes on an InferenceData object.

        Parameters
        ----------
        idata : arviz.InferenceData, optional
            The InferenceData object to set attributes on.

        Raises
        ------
        RuntimeError
            If no InferenceData object is provided.

        Returns
        -------
        None

        Examples
        --------
        >>> model = MyModel(ModelBuilder)
        >>> idata = az.InferenceData(your_dataset)
        >>> model.set_idata_attrs(idata=idata)
        >>> assert "id" in idata.attrs #this and the following lines are part of doctest, not user manual
        >>> assert "model_type" in idata.attrs
        >>> assert "version" in idata.attrs
        >>> assert "sampler_config" in idata.attrs
        >>> assert "model_config" in idata.attrs
        """
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        # Only classes with non-dataset parameters will implement save_input_params
        if hasattr(self, "_save_input_params"):
            self._save_input_params(idata)
        return idata
    
    @property
    def id(self) -> str:
        """Returns the model id."""
        hasher = hashlib.md5()
        hasher.update(str(self.version).encode())
        hasher.update(str(self._model_type).encode())
        return hasher.hexdigest()[:16]
    
    @abstractmethod
    def _extract_chains(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the chains from the model trace.

        Returns:
        ---------
        mu_chain: np.ndarray
            The chain of the mu parameter.
        sigma_chain: np.ndarray
            The chain of the sigma parameter.
        """
        raise NotImplementedError
    
    @abstractmethod
    def analyse_chains(self) -> Dict:
        """
        Analyse the chains of the model.

        Args:
            idata (az.InferenceData): The ArviZ InferenceData object containing the model trace.

        Returns:
            Dict: A dictionary containing the chain analysis.
        """
        
        raise NotImplementedError
    
    
    
    
    
    
    
    
    
    
    

