import random

import numpy as np


def set_seed(random_state: int = 42) -> None:
    """Function fixes random state to ensure results are reproducible"""
    np.random.seed(random_state)
    random.seed(random_state)
