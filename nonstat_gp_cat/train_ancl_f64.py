"""
Train A_N_C_L configuration with float64 precision to avoid numerical instability.
Run from nonstat_gp_cat directory.
"""
import torch
# Set default dtype to float64 BEFORE other imports
torch.set_default_dtype(torch.float64)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

from data_loader import Data
from nsgp_sgd import NSGP

# AQI breakpoints for India
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

def train_nsgp(dataloader, kernel='matern', time_kernel='local_per', iters=30, restarts=2):
    """Train NSGP model with multiple restarts."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, y = dataloader.train_X.to(torch.float64), dataloader.train_y.to(torch.float64)
    
    print(dataloader.train_df.head())
    
    best_model = None
    best_loss = float('inf')
    
    for restart in range(restarts):
        model = NSGP(
            train_x=X,
            train_y=y,
            cat_idx=dataloader.cat_idx,
            num_classes=dataloader.num_classes,
            space_kernel=kernel,
            time_kernel=time_kernel,
            jitter=1e-4
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        batch_size = min(2048, len(X))
        n_batches = len(X) // batch_size
        
        valid_training = True
        for i in range(iters):
            epoch_loss = 0
            for j in range(n_batches):
                idx = torch.randperm(len(X))[:batch_size]
                X_batch = X[idx].to(device)
                y_batch = y[idx].to(device)
                
                optimizer.zero_grad()
                loss = model(X_batch, y_batch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    Restart {restart+1}, Iter {i+1}: NaN/Inf loss detected, breaking")
                    valid_training = False
                    break
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            if not valid_training:
                break
            
            if (i + 1) % 10 == 0:
                print(f"    Restart {restart+1}, Iter {i+1}/{iters}, Loss: {epoch_loss/n_batches:.4f}", flush=True)
        
        if valid_training:
            final_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
            if final_loss < best_loss and not np.isinf(final_loss):
                best_loss = final_loss
                best_model = model
                print(f"    New best loss: {best_loss:.4f}", flush=True)
    
    return best_model

def predict_nsgp(model, test_X, y_mean, device='cuda'):
    """Make predictions with NSGP model."""
    model.eval()
    test_X = test_X.to(torch.float64).to(device)
    
    with torch.no_grad():
        pred_mean, pred_var = model.predict(test_X)
    
    pred_mean = pred_mean.cpu().numpy() + y_mean
    pred_var = pred_var.cpu().numpy()
    
    return pred_mean, pred_var

def main():
    print("=" * 80, flush=True)
    print("Running A_N_C_L Config with Float64 Precision", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Default dtype: {torch.get_default_dtype()}", flush=True)
    
    results_path = '../experiments/results/'
    os.makedirs(results_path, exist_ok=True)
    
    # A_N_C_L config: matern + local_per
    config = {
        'name': 'A_N_C_L',
        'latex': 'ANCL',
        'kernel': 'matern',
        'time_kernel': 'local_per'
    }
    
    print(f"\n{'='*80}", flush=True)
    print(f"Config: {config['name']} ({config['latex']})", flush=True)
    print(f"{'='*80}", flush=True)
    
    all_results = []
    
    for fold in range(3):
        print(f"\n  Fold {fold}:", flush=True)
        
        # Load data
        dataloader = Data(
            fold=fold,
            Xcols='longitude@latitude@humidity@temperature@weather@wind_direction@wind_speed@delta_t'
        )
        
        X = dataloader.train_X.to(torch.float64)
        y = dataloader.train_y.to(torch.float64)
        test_X = dataloader.test_X.to(torch.float64)
        test_y = dataloader.test_y.numpy()
        
        print(f"    Train: {X.shape}, Test: {test_X.shape}", flush=True)
        print(f"    Training NSGP (kernel={config['kernel']}, time={config['time_kernel']})...", flush=True)
        
        model = train_nsgp(dataloader, kernel=config['kernel'], 
                          time_kernel=config['time_kernel'], iters=30, restarts=2)
        
        if model is None:
            print("    FAILED - skipping", flush=True)
            continue
        
        # Predict
        print(dataloader.train_df.head(), flush=True)
        pred_y, pred_var = predict_nsgp(model, test_X, dataloader.y_mean, device)
        
        # Compute metrics
        y_true = test_y + dataloader.y_mean
        rmse = np.sqrt(mean_squared_error(y_true, pred_y))
        
        true_cats = [pm25_to_aqi_category(y) for y in y_true]
        pred_cats = [pm25_to_aqi_category(y) for y in pred_y]
        cat_acc = sum(1 for t, p in zip(true_cats, pred_cats) if t == p) / len(true_cats)
        
        print(f"    RMSE: {rmse:.2f}, Cat.Acc: {cat_acc*100:.1f}%", flush=True)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': pred_y,
            'y_var': pred_var,
            'true_cat': true_cats,
            'pred_cat': pred_cats
        })
        pred_df.to_csv(f"{results_path}{config['name']}_fold{fold}.csv", index=False)
        
        all_results.append({
            'config': config['name'],
            'fold': fold,
            'rmse': rmse,
            'cat_acc': cat_acc
        })
    
    # Summary
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*80}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        avg_rmse = df['rmse'].mean()
        avg_cat = df['cat_acc'].mean()
        print(f"A_N_C_L: Avg RMSE = {avg_rmse:.2f}, Avg Cat.Acc = {avg_cat*100:.1f}%", flush=True)
        print(f"Paper reports: RMSE = 23.51", flush=True)

if __name__ == '__main__':
    main()
