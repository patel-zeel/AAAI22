# !pip install gpytorch
# !pip install tqdm
#### !git clone https://github.com/Bjarten/early-stopping-pytorch.git
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import trange
# %cd 
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
from model import ExactGPModel

def get_loss(model, train_x, train_y, random_state=0):
    
    torch.manual_seed(random_state)
    model.eval()
    model.likelihood.eval()
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(train_x)
    loss = -mll(output, train_y)
    return loss.item()


def predict(model, test_x):
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    return observed_pred

def run_model(kernel, ard):
    
    if kernel =='rbf':
        gpytorch_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard)
    elif kernel =='matern12':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=ard)
    elif kernel =='matern32':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=ard)
    elif kernel =='matern52':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=ard)
    elif kernel =='matern_rbf':
        gpytorch_kernel = gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel =='maternXrbf':
        gpytorch_kernel = gpytorch.kernels.RBFKernel()*gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel =='local_p_delta':
        gpytorch_kernel = gpytorch.kernels.PeriodicKernel(active_dims=[7])
    else:
        print("Kernel Not Found")
        exit()
    
    return gpytorch_kernel

# if(__name__ == "main"):
    
fold = int(sys.argv[1])
kernel = str(sys.argv[2])
best_model_path = str(sys.argv[3])
data_str = str(sys.argv[4])
if(data_str == "march"):
    str_data = "_mar"
elif(data_str == "march_nsgp"):
    str_data = "_mar_nsgp"
else:
    str_data = ""

train_x,train_y,test_x,test_y,DELTA = return_data(fold=fold,type='time_feature',scale=True,data_str = str_data)

print("Fold: ",fold)
print("Data received")

ard = train_x.shape[1]

gpytorch_kernel = run_model(kernel, ard)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood, gpytorch_kernel).to(device)

model.load_state_dict(torch.load(best_model_path),strict=False)

observed_pred = predict(model, test_x) + DELTA
# print("RMSE: ", math.sqrt(mean_squared_error(test_y.flatten(), observed_pred.mean.cpu())))
# print([i for i in model.parameters()])
# print(vars(model.covar_module.base_kernel))

# print("RMSE: ", math.sqrt(mean_squared_error(test_y.flatten(), observed_pred.mean.cpu())))

# ---> MEAN PREDICTION
test_input = pd.read_csv('../data/'+'time_feature'+'/fold'+str(fold)+'/test_data' + str_data+'.csv.gz')
train_input = pd.read_csv('../data/'+'time_feature'+'/fold'+str(fold)+'/train_data' + str_data+'.csv.gz')
test_pred =[]
for i in test_input['delta_t']: 
    test_pred.append(train_input[train_input.delta_t==i].PM25_Concentration.mean())

print("RMSE: ", math.sqrt(mean_squared_error(test_y.flatten(), np.array(test_pred))))

#---> Saving Preds
# test_input["pred_PM25"] = observed_pred.mean.cpu()
# test_input["pred_PM25_var"] = observed_pred.variance.detach().cpu()
# # print(test_input)
# print("Saving to: ", f"final_preds/{data_str}_data/{kernel}_fold_{fold}.csv.gz")
# test_input.to_csv(f"final_preds/{data_str}_data/{kernel}_fold_{fold}.csv.gz",index=False)
# TEST = pd.read_csv(f"final_preds/{data_str}_data/{kernel}_fold_{fold}.csv.gz")
# print(TEST)

# ---> Plotting Timeseries predictions
    # for station in test_input.station_id.unique():
# station = test_input.station_id[3]
# rows = test_input[test_input['station_id']==station].index
# plt.figure(figsize=(30,7))
# a = plt.plot(np.array(observed_pred.mean.cpu())[rows],c='r',label="Predicted")
#     # plt.xticks(range(len(np.array(test_input['time'])[rows])),np.array(test_input['time'])[rows],rotation='vertical')
# b = plt.plot(np.array(test_input['PM25_Concentration'])[rows],c='b',label="Ground Truth")
# c = plt.plot(100*np.array(test_input['filled'])[rows],c='g')
# # lower, upper = observed_pred.confidence_region()
# # d = plt.fill_between(range(len(np.array(test_input['time'])[rows])),
# #                 lower.detach().cpu()[rows],#(observed_pred.mean.cpu()[rows] - 2*observed_pred.variance.detach().cpu()[rows]),
# #                 upper.detach().cpu()[rows],#(observed_pred.mean.cpu()[rows] + 2*observed_pred.variance.detach().cpu()[rows]),
# #                 color='gray', alpha=0.8,label="Confidence-Region")
# # plt.legend((a,b,c,d),('predicted','true','filled','confidence-region'))
# plt.legend()
# plt.title("Station {} | Model: {} kernel".format(station,kernel))
# plt.xlabel("Hour")
# plt.ylabel("AQ Value")
# plt.savefig("try.png")

#---> Err vs VAR
# plt.figure(figsize=(30,7))
# plt.scatter((observed_pred.mean.cpu() - np.array(test_input['PM25_Concentration'])).abs(), observed_pred.variance.detach().cpu())
# plt.xlabel("Absolute Error")
# plt.ylabel("Predicted Variance")
# plt.savefig("err_var.png")