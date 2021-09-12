# !pip install gpytorch
# !pip install tqdm
#### !git clone https://github.com/Bjarten/early-stopping-pytorch.git
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import trange
from early_stopping_pytorch.pytorchtools import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from time import time
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys

from data import return_data
# from model import ExactGPModel
from model import MixedSingleTaskGP

def train_model(model, train_x, train_y, epochs=200, warm_up=5, lr=0.1, random_state=0):
    
    torch.manual_seed(random_state)
    
    for params in model.parameters():
        torch.nn.init.normal_(params)
    
    early_stopping = EarlyStopping(patience=warm_up)
    
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    t = trange(epochs)
#     losses = []
    for i in t: # Zero gradients from previous iteration
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
#         losses.append(loss.item())
        loss.backward()
        # t.set_description('Iteration %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, epochs, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()), refresh=True)
        optimizer.step()
        if(i>50):
            early_stopping(loss, model)
            if early_stopping.early_stop:
                    print("Early stopping")
                    break
#     model.losses = losses
    return loss.item()

def predict(model, test_x):
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    return observed_pred
# if(__name__ == "main"):
    
fold = int(sys.argv[1])
kernel = str(sys.argv[2])
random_state = int(sys.argv[3])
data_str = str(sys.argv[4])
if(data_str == "march"):
    str_data = "_mar"
elif(data_str == "march_nsgp"):
    str_data = "_mar_nsgp"
else:
    str_data = ""

train_x,train_y,test_x,test_y, cat_indices = return_data(fold=fold,type='time_feature',scale=True,data_str = str_data)

print("Fold: ",fold)
print("Random State: ", random_state)
# ard = train_x.shape[1]
# gpytorch_kernel = run_model(kernel, ard)
def cont_kernel_factory(batch_shape, ard_num_dims, active_dims):    
    if kernel =='rbf':
        gpytorch_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
    elif kernel =='matern12':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=0.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
    elif kernel =='matern32':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=1.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
    elif kernel =='matern52':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
    elif kernel =='matern_rbf':
        gpytorch_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims) + gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
    elif kernel =='maternXrbf':
        gpytorch_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)*gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
    else:
        print("Kernel Not Found")
        exit()
    return gpytorch_kernel

likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood, gpytorch_kernel).to(device)
cat_dims = cat_indices
# print(train_y.unsqueeze(1).shape)
model = MixedSingleTaskGP(train_X=train_x,train_Y=train_y.unsqueeze(1), cat_dims=cat_dims, cont_kernel_factory=cont_kernel_factory, likelihood=likelihood).to(device)
# print(model)
# print("kernel",model.covar_module)

kwargs= {'epochs':400, 'warm_up':5, 'lr':0.05, 'random_state':random_state}
loss = train_model(model, train_x, train_y, **kwargs)
observed_pred = predict(model, test_x)
print("Loss: ", loss)
print("RMSE: ", math.sqrt(mean_squared_error(test_y, observed_pred.mean.cpu())))

torch.save(model.state_dict(), f"final_models/{data_str}_data/{kernel}_fold_{fold}_random_state_{random_state}")

