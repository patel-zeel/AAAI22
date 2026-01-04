"""
Run ALL NSGP variants (ANCL configurations) and compute AQI metrics.
This trains and tests all model configurations from the paper.
"""

import os
import sys
sys.path.insert(0, '/home/t-zeelpatel/zeel_thesis/AAAI22/nonstat_gp_cat')

import numpy as np
import pandas as pd
import torch
import gpytorch
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

from config import Config
from data_loader import Data
from nsgp_sgd import NSGP
from nsgp.utils.inducing_functions import f_kmeans

import warnings
warnings.filterwarnings('ignore')

# AQI computation functions
INDIA_AQI_BREAKPOINTS = [
    (0, 30, 0, 50, 'Good'),
    (31, 60, 51, 100, 'Satisfactory'),
    (61, 90, 101, 200, 'Moderate'),
    (91, 120, 201, 300, 'Poor'),
    (121, 250, 301, 400, 'Very Poor'),
    (251, 500, 401, 500, 'Severe'),
]

def pm25_to_aqi_category(pm25):
    for c_lo, c_hi, i_lo, i_hi, category in INDIA_AQI_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return category
    if pm25 > 500:
        return INDIA_AQI_BREAKPOINTS[-1][4]
    return INDIA_AQI_BREAKPOINTS[0][4]

def compute_aqi_metrics(y_true, y_pred):
    true_categories = [pm25_to_aqi_category(y) for y in y_true]
    pred_categories = [pm25_to_aqi_category(y) for y in y_pred]
    
    correct = sum(1 for t, p in zip(true_categories, pred_categories) if t == p)
    category_accuracy = correct / len(true_categories)
    
    categories = [bp[4] for bp in INDIA_AQI_BREAKPOINTS]
    off_by_one_correct = 0
    for t, p in zip(true_categories, pred_categories):
        t_idx = categories.index(t)
        p_idx = categories.index(p)
        if abs(t_idx - p_idx) <= 1:
            off_by_one_correct += 1
    off_by_one_accuracy = off_by_one_correct / len(true_categories)
    
    threshold = 90.0
    true_above = np.array(y_true) > threshold
    pred_above = np.array(y_pred) > threshold
    threshold_accuracy = np.mean(true_above == pred_above)
    
    return {
        'category_accuracy': category_accuracy,
        'off_by_one_accuracy': off_by_one_accuracy,
        'threshold_90_accuracy': threshold_accuracy,
    }


class ExactGPModel(gpytorch.models.ExactGP):
    """Standard GP model for baseline."""
    def __init__(self, train_x, train_y, likelihood, ard=True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_standard_gp(X_train, y_train, y_mean, ard=True, iters=100):
    """Train standard stationary GP. y_train should be centered."""
    X = X_train.to(Config.device)
    y = y_train.to(Config.device).ravel()
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(Config.device)
    model = ExactGPModel(X, y, likelihood, ard=ard).to(Config.device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(iters):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f"    Iter {i+1}/{iters}, Loss: {loss.item():.4f}")
    
    return model, likelihood


def predict_standard_gp(model, likelihood, X_test, y_mean):
    """Predict with standard GP. Returns predictions in original scale."""
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        X_test = X_test.to(Config.device)
        pred = likelihood(model(X_test))
        # Add back y_mean to get original scale
        return pred.mean.cpu().numpy() + y_mean, pred.variance.cpu().numpy()


def train_nsgp(dataloader, kernel='rbf', time_kernel='rbf', iters=40, restarts=3):
    """Train NSGP model."""
    X, y, Xcols = dataloader.load_train()
    
    # Create inducing points for each dimension
    X_bar = []
    Num_Ind = 100
    for dim in range(X.shape[1]):
        if X[:, dim].unique().shape[0] < Num_Ind:
            X_bar.append(X[:, dim].unique().reshape(-1, 1).to(Config.device))
        else:
            X_bar.append(f_kmeans(X[:, dim:dim+1], num_inducing_points=Num_Ind, random_state=0).to(Config.device))
    
    train = TensorDataset(X, y)
    trainloader = DataLoader(train, batch_size=X.shape[0]//4, shuffle=True)
    
    best_model = None
    best_loss = np.inf
    
    for restart in range(restarts):
        torch.manual_seed(restart)
        try:
            model = NSGP(X_bar, dataloader.cat_indicator, dataloader.time_indicator,
                        random_state=restart, jitter=1e-3, use_Trans=False, m=3,
                        kernel=kernel, timekernel=time_kernel).to(Config.device)
        except Exception as e:
            print(f"    Failed to create NSGP model: {e}")
            return None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # Initialize parameters
        with torch.no_grad():
            if time_kernel in ['loc_periodic', 'periodic', 'local_per'] and hasattr(model, 'period'):
                torch.nn.init.uniform_(model.period, a=0.03, b=0.07)
            torch.nn.init.uniform_(model.local_gp_ls, a=0.5, b=3)
            torch.nn.init.uniform_(model.local_gp_std, a=0.5, b=3)
            torch.nn.init.uniform_(model.local_gp_noise_std, a=0.5, b=3)
            for param in model.local_ls:
                torch.nn.init.uniform_(param, a=0.5, b=3)
            torch.nn.init.uniform_(model.global_gp_std, a=0.5, b=3)
            torch.nn.init.uniform_(model.global_gp_noise_std, a=0.5, b=3)
        
        final_loss = np.inf
        for i in range(iters):
            loss_val = 0
            for X_batch, y_batch in trainloader:
                optimizer.zero_grad()
                try:
                    loss = model(X_batch.to(Config.device), y_batch.to(Config.device))
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for param in model.parameters():
                            torch.clamp_(param, min=1e-5)
                    loss_val += loss.item()
                except Exception as e:
                    loss_val = np.inf
                    break
            
            final_loss = loss_val / len(trainloader)
            if (i + 1) % 10 == 0:
                print(f"    Restart {restart+1}, Iter {i+1}/{iters}, Loss: {final_loss:.4f}")
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            print(f"    New best loss: {best_loss:.4f}")
    
    return best_model


def predict_nsgp(model, dataloader):
    """Predict with NSGP model."""
    X, y, _ = dataloader.load_train()
    test_X, test_y = dataloader.load_test()
    
    model.eval()
    with torch.no_grad():
        pred_y, pred_var = model.predict(
            X.to(Config.device), y.to(Config.device), test_X.to(Config.device)
        )
    
    # Convert back to original scale
    pred_y = pred_y.cpu().numpy().ravel() + dataloader.y_mean
    pred_var = pred_var.diagonal().cpu().numpy()
    
    return pred_y, pred_var


# Model configurations
# Paper notation: A=ARD, N=NonStationary, C=Categorical, L=LocalPeriodic
# Bar means "off"
CONFIGS = {
    'A_bar_N_bar_C_bar_L_bar': {  # Baseline GP - no ARD, no nonstat, one-hot, RBF time
        'type': 'gp',
        'ard': False,
        'Xcols': 'one_hot',
        'latex': r'\bar{A}\bar{N}\bar{C}\bar{L}',
    },
    'A_N_bar_C_bar_L_bar': {  # ARD GP - ARD on, no nonstat, one-hot, RBF time
        'type': 'gp',
        'ard': True,
        'Xcols': 'one_hot',
        'latex': r'A\bar{N}\bar{C}\bar{L}',
    },
    'A_N_bar_C_L_bar': {  # ARD + Categorical - ARD on, no nonstat, cat, RBF time
        'type': 'nsgp',
        'kernel': 'rbf',
        'time_kernel': 'rbf',
        'Xcols': 'cat',
        'latex': r'A\bar{N}C\bar{L}',
    },
    'A_N_bar_C_L': {  # ARD + Categorical + LocalPeriodic (best by NLML)
        'type': 'nsgp',
        'kernel': 'rbf',
        'time_kernel': 'local_per',
        'Xcols': 'cat',
        'latex': r'A\bar{N}CL',
    },
    'A_N_C_L_bar': {  # All except LocalPeriodic
        'type': 'nsgp',
        'kernel': 'm32',  # non-stationary matern 3/2
        'time_kernel': 'rbf',
        'Xcols': 'cat',
        'latex': r'ANC\bar{L}',
    },
    'A_N_C_L': {  # All extensions on
        'type': 'nsgp',
        'kernel': 'rbf',  # RBF kernel
        'time_kernel': 'local_per',
        'Xcols': 'cat',
        'latex': r'ANCL',
    },
}


def main():
    print("=" * 80)
    print("Running NSGP Experiments for All ANCL Configurations")
    print("=" * 80)
    
    # Run all configs
    # configs_to_run = list(CONFIGS.keys())
    configs_to_run = ['A_N_C_L_bar']  # Run only missing config
    
    all_results = []
    
    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        print(f"\n{'='*80}")
        print(f"Config: {config_name} ({config['latex']})")
        print(f"{'='*80}")
        
        fold_results = []
        
        for fold in ['0', '1', '2']:
            print(f"\n  Fold {fold}:")
            
            if config['Xcols'] == 'cat':
                Xcols = 'longitude@latitude@humidity@temperature@weather@wind_direction@wind_speed@delta_t'
            else:
                # One-hot encoded (handled differently in data loader)
                Xcols = 'longitude@latitude@humidity@temperature@weather@wind_direction@wind_speed@delta_t'
            
            dataloader = Data(fold, Xcols=Xcols)
            X, y, _ = dataloader.load_train()
            test_X, test_y = dataloader.load_test()
            
            print(f"    Train: {X.shape}, Test: {test_X.shape}")
            
            if config['type'] == 'gp':
                print(f"    Training standard GP (ARD={config['ard']})...")
                model, likelihood = train_standard_gp(X, y, dataloader.y_mean, ard=config['ard'], iters=100)
                pred_y, pred_var = predict_standard_gp(model, likelihood, test_X, dataloader.y_mean)
                # pred_y already has y_mean added back
            else:
                print(f"    Training NSGP (kernel={config['kernel']}, time={config['time_kernel']})...")
                model = train_nsgp(dataloader, kernel=config['kernel'], 
                                  time_kernel=config['time_kernel'], iters=30, restarts=2)
                if model is None:
                    print(f"    FAILED - skipping")
                    continue
                pred_y, pred_var = predict_nsgp(model, dataloader)
                # pred_y already has y_mean added back in predict_nsgp
            
            # Get ground truth - test_y is NOT centered (already in original scale)
            y_true = test_y.cpu().numpy().ravel()
            
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_true, pred_y))
            mae = mean_absolute_error(y_true, pred_y)
            r2 = r2_score(y_true, pred_y)
            aqi = compute_aqi_metrics(y_true, pred_y)
            
            print(f"    RMSE: {rmse:.2f}, Cat.Acc: {aqi['category_accuracy']*100:.1f}%")
            
            fold_results.append({
                'fold': fold,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'category_accuracy': aqi['category_accuracy'],
                'off_by_one_accuracy': aqi['off_by_one_accuracy'],
                'threshold_90_accuracy': aqi['threshold_90_accuracy'],
            })
            
            # Save predictions
            save_path = Config.res_path + f"{config_name}_fold{fold}.csv"
            pd.DataFrame({
                'y_true': y_true,
                'y_pred': pred_y,
            }).to_csv(save_path, index=False)
        
        if fold_results:
            avg_rmse = np.mean([r['rmse'] for r in fold_results])
            avg_mae = np.mean([r['mae'] for r in fold_results])
            avg_r2 = np.mean([r['r2'] for r in fold_results])
            avg_cat = np.mean([r['category_accuracy'] for r in fold_results])
            avg_off1 = np.mean([r['off_by_one_accuracy'] for r in fold_results])
            avg_thresh = np.mean([r['threshold_90_accuracy'] for r in fold_results])
            
            all_results.append({
                'config': config_name,
                'latex': config['latex'],
                'rmse': avg_rmse,
                'mae': avg_mae,
                'r2': avg_r2,
                'category_accuracy': avg_cat,
                'off_by_one_accuracy': avg_off1,
                'threshold_90_accuracy': avg_thresh,
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        results_df.to_csv(Config.res_path + 'nsgp_ancl_results.csv', index=False)
        print(f"\nResults saved to: {Config.res_path}nsgp_ancl_results.csv")


if __name__ == '__main__':
    main()
