# !pip install gpytorch
# !pip install tqdm
#### !git clone https://github.com/Bjarten/early-stopping-pytorch.git
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import trange
# %cd 
# from early_stopping_pytorch.pytorchtools import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from time import time
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def return_data(fold,type,scale,data_str):
    train_input = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/train_data' + data_str+'.csv.gz')
    test_input = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/test_data' + data_str+'.csv.gz')
    test_output = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/test_output' + data_str+'.csv.gz')
    train_output = np.array(train_input['PM25_Concentration'])
    train_input= train_input.drop(['station_id','PM25_Concentration','time','filled'],axis=1).values
    try:
        test_input= test_input.drop(['PM25_Concentration','station_id','time','filled'],axis=1).values
    except:
        test_input= test_input.drop(['station_id','time'],axis=1).values
    test_output= test_output.drop(['time'],axis=1)
    if(scale==True):
        scaler_X = StandardScaler().fit(train_input)
        train_input = scaler_X.transform(train_input)
        test_input = scaler_X.transform(test_input)
        # scaler_y = StandardScaler().fit(train_output.reshape(-1, 1))
        DELTA = train_output.mean()
        train_output -= DELTA

    # return torch.Tensor(train_input).to(device),torch.Tensor(train_output).to(device),torch.Tensor(test_input).to(device),torch.Tensor(test_output.values).to(device)
    return torch.Tensor(train_input).to(device),torch.Tensor(train_output).to(device),torch.Tensor(test_input).to(device), test_output.values, DELTA
#     return torch.Tensor(train_input),torch.Tensor(train_output),torch.Tensor(test_input),torch.Tensor(test_output.values)
