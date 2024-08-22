"""This module contains project level constants"""

import secrets
import warnings

import numpy as np
from numpy.random import RandomState
from sklearn.utils._random import check_random_state

warnings.filterwarnings("ignore")

# The NumPy Generator will be used throughout the whole experiment
SEED: int = secrets.randbelow(exclusive_upper_bound=int(pow(2, 32) - 1))
np.random.seed(SEED)
rng: RandomState = check_random_state(SEED)
