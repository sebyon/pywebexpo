import numpy as np
import pymc as pm
import arviz as az

import utils as utils
from seg_models.informedvar import SEGInformedVarLognormal

from logging import Logger

logger = Logger(__name__)

# Import all the models from the seg_models module and the worker_models module



default_data = ['24.7', '64.1', '13.8', '43.7', '19.9', '133', '32.1', '15', '53.7']

error_data = [
    '96.6', '38.3', '80.8', '15.1', '34', '73.4', '14.5', '64.8', '27.4', '48.7',
    '43.3', '43.4', '57.8', '94.9', '44.1', '44.3', '62.9', '117', '51.6', '64.7',
    '50.1', '74.7', '221', '46.8', '84.3', '93.4', '126', '46.9', '29.5', '73.8',
    '66.9', '61.3', '30.2', '101', '22.6', '191', '29.3', '68', '114', '33.7',
    '52.5', '118', '49.7', '60.4', '36.6', '55.9', '31.9', '84.3', '75.8', '39.5',
    '28.3', '56.5', '44.2', '48', '36.6', '70', '37', '72', '48', '66.1',
    '72.4', '80.9', '69.1', '162', '67.3', '75.2', '40.5', '25.6', '44', '120',
    '56.3', '42.9', '6.63', '24.9', '40.9', '81', '97.2', '74.7', '79.6', '48.8',
    '75.3', '54.8', '66.5', '71.3', '28.7', '87.5', '51.9', '19.6', '60.8', '45.9',
    '46.9', '84.8', '120', '103', '36.7', '92.7', '32.8', '73.8', '214', '65.3'
]

analysis_config = {
    "probacred": 80,
    "frac_threshold": 10,
    "target_perc": 95
}

## Check if lognormal data is within limits of model
data_check = utils.check_lognormal_data(data = np.array(error_data).astype(float), oel = 100)
print(data_check)

model = SEGInformedVarLognormal(data = error_data, oel = 100, error_mode='CV', analysis_config=analysis_config)

# test if error mode is correctly set

model.fit()
model.analyse_chains()

print(model.analysis)

# Check divergences
divergences = utils.inspect_divergent_traces(model)
print(divergences)