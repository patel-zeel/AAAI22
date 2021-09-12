from config import Config
import numpy as np
import sys
import pandas as pd
import torch
from time import time
import warnings
warnings.filterwarnings('ignore')
from torch.cuda.amp import GradScaler, autocast
import torch.autograd.profiler as profiler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from snsgp import SNSGP
# from nsgp import NSGP
from nsgp_sgd import NSGP
from nsgp.utils.inducing_functions import f_kmeans, f_random
import matplotlib.pyplot as plt
from IPython.display import clear_output
torch.autograd.set_detect_anomaly(True)