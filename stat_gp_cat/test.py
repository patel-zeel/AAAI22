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
best_model_path = str(sys.argv[3])
data_str = str(sys.argv[4])

if(data_str == "march"):
    str_data = "_mar"
elif(data_str == "march_nsgp"):
    str_data = "_mar_nsgp"
else:
    str_data = ""

train_x,train_y,test_x,test_y, cat_indices = return_data(fold=fold,type='time_feature',scale=True,data_str = str_data)

print("Fold: ",fold)
# print("Random State: ", random_state)
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

model.load_state_dict(torch.load(best_model_path))
observed_pred = predict(model, test_x)
# print("RMSE: ", math.sqrt(mean_squared_error(test_y.cpu().flatten(), observed_pred.mean.cpu())))
print("RMSE: ", math.sqrt(mean_squared_error(test_y.flatten(), observed_pred.mean.cpu())))

test_input = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/test_data' + data_str+'.csv.gz')
test_output = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/test_output' + data_str+'.csv.gz')
test_output= test_output.drop(['time'],axis=1)
stationids = test_input.station_id.unique()
for station in stationids:
    rows = test_input[test_input['station_id']==station].index
    err = mean_squared_error(observed_pred.mean.cpu()[rows], np.array(test_output)[rows], squared=False)
    print(station,err)

# test_input = pd.read_csv('../data/time_feature'+'/fold'+str(fold)+'/test_data_'+'mar'+'.csv.gz')
#     # # for station in test_input.station_id.unique():
#     # station = 1030
#     # rows = test_input[test_input['station_id']==station].index
#     # plt.figure(figsize=(30,7))
#     # a = plt.plot(np.array(observed_pred.mean.cpu())[rows],c='r',label="Predicted")
#     #     # plt.xticks(range(len(np.array(test_input['time'])[rows])),np.array(test_input['time'])[rows],rotation='vertical')
#     # b = plt.plot(np.array(test_input['PM25_Concentration'])[rows],c='b',label="Ground Truth")
#     # c = plt.plot(100*np.array(test_input['filled'])[rows],c='g')
#     # lower, upper = observed_pred.confidence_region()
#     # d = plt.fill_between(range(len(np.array(test_input['time'])[rows])),
#     #                 lower.detach().cpu()[rows],#(observed_pred.mean.cpu()[rows] - 2*observed_pred.variance.detach().cpu()[rows]),
#     #                 upper.detach().cpu()[rows],#(observed_pred.mean.cpu()[rows] + 2*observed_pred.variance.detach().cpu()[rows]),
#     #                 color='gray', alpha=0.8,label="Confidence-Region")
#     # # plt.legend((a,b,c,d),('predicted','true','filled','confidence-region'))
#     # plt.legend()
#     # plt.title("Station {} | Model: {} kernel".format(station,kernel))
#     # plt.xlabel("Hour")
#     # plt.ylabel("AQ Value")
#     # plt.savefig("try.png")

# plt.figure(figsize=(30,7))
# plt.scatter((observed_pred.mean.cpu() - np.array(test_input['PM25_Concentration'])).abs(), observed_pred.variance.detach().cpu())
# plt.xlabel("Absolute Error")
# plt.ylabel("Predicted Variance")
# plt.savefig("err_var.png")