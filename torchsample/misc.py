from enum import Enum, auto
import random
import numpy as np
import torch

class ExecType(Enum):
    TRAIN = auto()
    VAL = auto()

# from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def trun_n_d(n,d):
    '''
    Truncate float (n) to (d) decimal places
    :param n: float to truncate
    :param d: how many decimal places to truncate to
    :return:
    '''
    return int(n*10**d)/10**d

def is_iterable(x):
    return isinstance(x, (tuple, list))
def is_tuple_or_list(x):
    return isinstance(x, (tuple, list))


def time_left_str(seconds):
    '''
    Produces a human-readable string in hh:mm:ss format
    :param seconds:

    :return:
    '''
    # seconds = 370000.0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        thetime = "Projected time remaining  |  {:d}d:{:d}h:{:02d}m".format(d, h, m)
    elif h > 0:
        thetime = "Projected time remaining:  |  {:d}h:{:02d}m".format(h, m)
    elif m > 0:
        thetime = "Projected time remaining:  |  {:02d}m:{:02d}s".format(m, s)
    else:
        thetime = "Projected time remaining:  |  {:02d}s".format(seconds)
    return thetime


def initialize_random(seed, init_cuda=True):
    '''
    Initializes random seed for all aspects of training: python, numpy, torch, cuda

    :param seed:
    :param init_cuda: whether to init cuda seed
    :return:
    '''
    ## Initialize random seed for repeatability ##
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if init_cuda:
        torch.cuda.manual_seed_all(seed)
    ## END ##