import numpy as np
import os
import pickle
import random


def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_trace(path, loss):
    if not os.path.isfile(path):
        return None
    f = open(path, 'rb')
    trace = pickle.load(f)
    trace.loss = loss
    f.close()
    return trace