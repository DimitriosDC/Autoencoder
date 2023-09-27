from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import math
import pandas as pd
import torch
import torchvision
from torch import nn, optim
from sklearn import decomposition
import statistics
import torch.nn.functional as F

import os
import sys

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import sys
seed =  int(sys.argv[1])

torch.manual_seed(seed) # torch seed
random.seed(seed) # Removed this to set seed globally for whole file
np.random.seed(seed)

print(seed*2)
