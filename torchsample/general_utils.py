import numpy as np
import os
import random
import torch
from torch import nn

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def initialize_random(seed):
    ## Initialize random seed for repeatability ##
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)