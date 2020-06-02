import math
import numpy as np
import os
import pickle
import random


def relative_round(x):
    """
    A util that rounds the input to the most significant digits.
    Useful for storing the results as rounding float
    numbers may cause file name ambiguity.
    """
    mantissa, exponent = math.frexp(x)
    return round(mantissa, 3) * 2**exponent

    
def get_trace(path, loss):
    if not os.path.isfile(path):
        return None
    f = open(path, 'rb')
    trace = pickle.load(f)
    trace.loss = loss
    f.close()
    return trace


def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
