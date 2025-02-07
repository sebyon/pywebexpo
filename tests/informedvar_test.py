import unittest
import numpy as np
import pymc as pm
from pywebexpo.seg_models.informedvar import SEGInformedVarLognormal

class TestSEGInformedVarLognormal(unittest.TestCase):

    def setUp(self):
        self.data = ['24.7', '64.1', '13.8', '43.7', '19.9', '133', '32.1', '15', '53.7']
        self.oel = 100
        self.model = SEGInformedVarLognormal(data=self.data, oel=self.oel)

    def test_initialization(self):
        self.assertEqual(self.model.data, self.data)
        self.assertEqual(self.model.oel, self.oel)
        self.assertIsNotNone(self.model.sampler_config)
        self.assertIsNotNone(self.model.model_config)
        self.assertIsNotNone(self.model.error_mode)
        self.assertIsNotNone(self.model.analysis_config)

    def test_build_model(self):
        pymc_model = self.model.build_model()
        self.assertIsInstance(pymc_model, pm.Model)

    def test_sample_model(self):
        self.model.build_model()
        self.model.sample_model(draws=5000, tune=5000, chains=4)
        self.assertIsNotNone(self.model.idata)

    def test_fit(self):
        pymc_model = self.model.fit()
        self.assertIsInstance(pymc_model, pm.Model)
        self.assertIsNotNone(self.model.idata)

    def test_analyse_chains(self):
        self.model.fit()
        analysis = self.model.analyse_chains()
        self.assertIsNotNone(analysis)

    def test_get_default_error_mode(self):
        self.assertIsNone(self.model.get_default_error_mode())

    def test_get_default_model_config(self):
        model_config = self.model.get_default_model_config()
        self.assertIsInstance(model_config, dict)
        self.assertIn("mu_lower", model_config)
        self.assertIn("mu_upper", model_config)

    def test_get_default_sample_config(self):
        sample_config = self.model.get_default_sample_config()
        self.assertIsInstance(sample_config, dict)
        self.assertIn("draws", sample_config)
        self.assertIn("tune", sample_config)

    def test_get_default_analysis_config(self):
        analysis_config = self.model.get_default_analysis_config()
        self.assertIsInstance(analysis_config, dict)
        self.assertIn("probacred", analysis_config)
        self.assertIn("frac_threshold", analysis_config)

    def test_generate_and_preprocess_model_data(self):
        self.model._generate_and_preprocess_model_data()
        self.assertIsNotNone(self.model.y_data)
        self.assertIsNotNone(self.model.lower_limits)
        self.assertIsNotNone(self.model.upper_limits)

    def test_extract_chains(self):
        self.model.fit()
        mu_chain, sigma_chain = self.model._extract_chains()
        self.assertIsInstance(mu_chain, np.ndarray)
        self.assertIsInstance(sigma_chain, np.ndarray)

if __name__ == "__main__":
    unittest.main()