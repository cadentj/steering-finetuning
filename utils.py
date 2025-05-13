import random
import torch as t
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    np.random.seed(seed)