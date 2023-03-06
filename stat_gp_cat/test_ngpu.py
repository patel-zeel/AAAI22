# !pip install gpytorch
# !pip install tqdm
#### !git clone https://github.com/Bjarten/early-stopping-pytorch.git
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import trange
from pytorchtools import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from time import time
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import sys
import gc
from data import return_data
# from model import ExactGPModel
from model_ngpu import MixedSingleTaskGP
from latexify import latexify

def train_model(model, 
                train_x, train_y, 
                epochs=200, warm_up=5, lr=0.1, random_state=0, 
                checkpoint_size=10000,
                preconditioner_size=100,):
    
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
    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):
        
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


def find_best_gpu_setting(model,
                          train_x,
                          train_y,
                          n_devices,
                          kwargs,
                          preconditioner_size
):
    N = train_x.size(0)

    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            # _, _ = train(train_x, train_y,
            #              n_devices=n_devices, output_device=output_device,
            #              checkpoint_size=checkpoint_size,
            #              preconditioner_size=preconditioner_size, n_training_iter=1)

            _ = train_model(model, train_x, train_y, **kwargs, checkpoint_size=checkpoint_size, preconditioner_size=preconditioner_size)

            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size



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

train_x,train_y,test_x,test_y, cat_indices, DELTA = return_data(fold=fold,type='time_feature',scale=True,data_str = str_data)

print("Fold: ",fold)
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
    elif kernel =='local_p_matern12':
        gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=0.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)*gpytorch.kernels.PeriodicKernel(active_dims=[7])
    elif kernel =='local_p_with_rbf':
        cont = gpytorch.kernels.RBFKernel(batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
        lp = gpytorch.kernels.RBFKernel(active_dims=[7])*gpytorch.kernels.PeriodicKernel(active_dims=[7])
        gpytorch_kernel = cont*lp
    elif kernel =='local_p_with_matern12':
        cont = gpytorch.kernels.MaternKernel(nu=0.5,batch_shape=batch_shape,ard_num_dims=ard_num_dims,active_dims=active_dims)
        lp = gpytorch.kernels.RBFKernel(active_dims=[7])*gpytorch.kernels.PeriodicKernel(active_dims=[7])
        gpytorch_kernel = cont*lp
    else:
        print("Kernel Not Found")
        exit()
    return gpytorch_kernel

likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood, gpytorch_kernel).to(device)
cat_dims = cat_indices
# print(train_y.unsqueeze(1).shape)
n_devices=2
model = MixedSingleTaskGP(train_X=train_x,train_Y=train_y.unsqueeze(1), cat_dims=cat_dims, cont_kernel_factory=cont_kernel_factory, likelihood=likelihood, output_device=device, n_devices=n_devices).to(device)

model.load_state_dict(torch.load(best_model_path),strict=False)
observed_pred = predict(model, test_x) + DELTA
# print("RMSE: ", math.sqrt(mean_squared_error(test_y.cpu().flatten(), observed_pred.mean.cpu())))
print("RMSE: ", math.sqrt(mean_squared_error(test_y.flatten(), observed_pred.mean.cpu())))

# print("Matern_Kernel")
# M_length_scales = list(model.covar_module.base_kernel.base_kernel.kernels[0].raw_lengthscale.cpu().detach().numpy()[0])
# M_active_dims = list(model.covar_module.base_kernel.base_kernel.kernels[0].active_dims.cpu().numpy())
# # print(M_length_scales)
# # print(M_active_dims)
# print(dict(zip(M_active_dims,M_length_scales)))

# print("CategoricalKernel")
# # print(vars(model.covar_module.base_kernel.base_kernel.kernels[1]))
# C_length_scales = list(model.covar_module.base_kernel.base_kernel.kernels[1].raw_lengthscale.cpu().detach().numpy()[0])
# C_active_dims = list(model.covar_module.base_kernel.base_kernel.kernels[1].active_dims.cpu().numpy())

# print(dict(zip(C_active_dims,C_length_scales)))




test_input = pd.read_csv('../data/'+'time_feature'+'/fold'+str(fold)+'/test_data' + str_data+'.csv.gz')
# print(list(test_input.columns))
test_input["pred_PM25"] = observed_pred.mean.cpu()
test_input["pred_PM25_var"] = observed_pred.variance.detach().cpu()
print("Saving to: ", f"final_preds/{data_str}_data/{kernel}_fold_{fold}.csv.gz")
test_input.to_csv(f"final_preds/{data_str}_data_new/{kernel}_fold_{fold}.csv.gz",index=False)
# TEST = pd.read_csv(f"final_preds/{data_str}_data/{kernel}_fold_{fold}.csv.gz")
# print(TEST)


# FOR COMPLETE PLOT
# test_input = pd.read_csv('../data/time_feature'+'/fold'+str(fold)+'/test_data_'+'mar_nsgp'+'.csv.gz')
# # for station in test_input.station_id.unique():
# station = test_input.station_id.unique()[-5]
# rows = test_input[test_input['station_id']==station].index
# rows = rows

# latexify(fig_width=2*3.32)
# ax = plt.figure(figsize=(30,7)).subplots(3,1)
# a = ax[0].plot(np.array(observed_pred.mean.cpu())[rows],c='r',label="Predicted")
#     # ax[0].xticks(range(len(np.array(test_input['time'])[rows])),np.array(test_input['time'])[rows],rotation='vertical')
# b = ax[0].plot(np.array(test_input['PM25_Concentration'])[rows],c='b',label="Ground Truth")
# # c = ax[0].plot(100*np.array(test_input['filled'])[rows],c='g')
# # lower, upper = observed_pred.confidence_region()
# # d = ax[0].fill_between(range(len(np.array(test_input['time'])[rows])),
# #                 # lower.detach().cpu()[rows],
# #                 (observed_pred.mean.cpu()[rows] - observed_pred.variance.detach().cpu()[rows]),
# #                 # upper.detach().cpu()[rows],
# #                 (observed_pred.mean.cpu()[rows] + observed_pred.variance.detach().cpu()[rows]),
# #                 color='gray', alpha=0.8,label="Confidence-Region")
# # ax[0].legend()
# ax[0].set_title("Station {} | Model: {} kernel".format(station,kernel))
# ax[0].set_xlabel("Hour")
# ax[0].set_ylabel("AQ Value")
# ax[0].set_ylim(0,350)
# # plt.savefig("sgp.png")

# XGB = pd.read_csv('../ml_results/fold0/XGB_scaled.csv.gz')
# # plt.figure(figsize=(30,7))
# a = ax[1].plot(np.array(XGB.prediction)[rows],c='r',label="Predicted")
# b = ax[1].plot(np.array(test_input['PM25_Concentration'])[rows],c='b',label="Ground Truth")
# ax[1].legend()
# ax[1].set_title("Station {} | Model: XGBoost".format(station))
# ax[1].set_xlabel("Hour")
# ax[1].set_ylabel("AQ Value")
# ax[1].set_ylim(0,350)
# # plt.savefig("xgb.png")

# RF = pd.read_csv('../ml_results/fold0/RF_scaled.csv.gz')
# # plt.figure(figsize=(30,7))
# a = ax[2].plot(np.array(RF.prediction)[rows],c='r',label="Predicted")
# b = ax[2].plot(np.array(test_input['PM25_Concentration'])[rows],c='b',label="Ground Truth")
# # ax[2].legend()
# ax[2].set_title("Station {} | Model: Random Forest".format(station))
# ax[2].set_xlabel("Hour")
# ax[2].set_ylabel("AQ Value")
# ax[2].set_ylim(0,350)

# plt.tight_layout()
# format_axes(ax[0])
# format_axes(ax[1])
# format_axes(ax[2])

# plt.savefig(f"comp_{station}.png")







# plt.figure(figsize=(30,7))

# df = pd.DataFrame()
# df['err'] = (observed_pred.mean.cpu() - np.array(test_input['PM25_Concentration'])).abs()
# df['var'] = observed_pred.variance.detach().cpu()
# df.to_csv(f"best_gp_{fold}.csv.gz",index=False)

# latexify(fig_width=3.32)
# plt.scatter((observed_pred.mean.cpu() - np.array(test_input['PM25_Concentration'])).abs(), observed_pred.variance.detach().cpu())
# plt.xlabel("Absolute Error")
# plt.ylabel("Predicted Variance")
# plt.title("Error vs Variance Plot for best Stationary GP method")
# plt.tight_layout()
# ax = plt.axes()
# format_axes(ax)
# plt.savefig("err_var.png")


# # FOR STATION-WISE RMSE
# test_input = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/test_data' + data_str+'.csv.gz')
# test_output = pd.read_csv('../data/'+type+'/fold'+str(fold)+'/test_output' + data_str+'.csv.gz')
# test_output= test_output.drop(['time'],axis=1)
# stationids = test_input.station_id.unique()
# for station in stationids:
#     rows = test_input[test_input['station_id']==station].index
#     err = mean_squared_error(observed_pred.mean.cpu()[rows], np.array(test_output)[rows], squared=False)
#     print(station,err)