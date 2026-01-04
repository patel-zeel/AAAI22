"""
Run A_N_C_L configuration with float64 precision to avoid numerical instability.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set float64 as default
torch.set_default_dtype(torch.float64)

from nonstat_gp_cat.data_loader import Data
from nonstat_gp_cat.nsgp_sgd import NSGP_ELBO_SGD

# AQI breakpoints
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
    true_cat = [pm25_to_aqi_category(y) for y in y_true]
    pred_cat = [pm25_to_aqi_category(y) for y in y_pred]
    cat_acc = sum(1 for t, p in zip(true_cat, pred_cat) if t == p) / len(true_cat)
    
    categories = [bp[4] for bp in INDIA_AQI_BREAKPOINTS]
    off_by_one = sum(1 for t, p in zip(true_cat, pred_cat) 
                    if abs(categories.index(t) - categories.index(p)) <= 1) / len(true_cat)
    
    threshold_90 = np.mean((np.array(y_true) > 90) == (np.array(y_pred) > 90))
    
    return cat_acc, off_by_one, threshold_90

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_inducing = 200
    batch_size = 256
    jitter = 1e-6

def train_nsgp(dataloader, kernel='matern', time_kernel='local_per', iters=50, restarts=2):
    """Train NSGP with multiple restarts."""
    X, y = dataloader.train_X, dataloader.train_y
    Xm = dataloader.Xm
    
    # Convert to float64
    X = X.double()
    y = y.double()
    Xm = Xm.double()
    
    print(f"    Data dtype: X={X.dtype}, y={y.dtype}, Xm={Xm.dtype}")
    
    best_model = None
    best_loss = float('inf')
    
    for restart in range(restarts):
        model = NSGP_ELBO_SGD(
            X, y, Xm,
            n_inducing=Config.n_inducing,
            kernel=kernel,
            time_kernel=time_kernel,
            jitter=Config.jitter
        ).to(Config.device)
        
        # Convert model parameters to float64
        model = model.double()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
        
        for epoch in range(iters):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = model(X_batch.to(Config.device), y_batch.to(Config.device))
                if torch.isnan(loss) or torch.isinf(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"    Restart {restart+1}, Iter {epoch+1}/{iters}, Loss: {avg_loss:.4f}")
        
        final_loss = epoch_loss / len(loader)
        if final_loss < best_loss and not np.isinf(final_loss):
            best_loss = final_loss
            best_model = model
            print(f"    New best loss: {best_loss:.4f}")
    
    return best_model

def predict_nsgp(model, X_test, y_mean):
    """Make predictions with NSGP."""
    model.eval()
    X_test = X_test.double().to(Config.device)
    
    with torch.no_grad():
        pred_mean, pred_var = model.predict(X_test)
        pred_mean = pred_mean.cpu().numpy() + y_mean
        pred_var = pred_var.cpu().numpy()
    
    return pred_mean, pred_var

def main():
    print("=" * 80)
    print("Running A_N_C_L Configuration with Float64 Precision")
    print("=" * 80)
    
    results_path = 'experiments/results/'
    os.makedirs(results_path, exist_ok=True)
    
    config_name = 'A_N_C_L'
    all_results = []
    
    for fold in range(3):
        print(f"\n  Fold {fold}:")
        
        # Load data
        dataloader = Data(
            path='data/delhi_aqi_dataset.csv',
            fold=fold,
            extra_days=0,
        )
        
        X, y = dataloader.train_X, dataloader.train_y
        test_X, test_y = dataloader.test_X, dataloader.test_y
        
        print(f"    Train: {X.shape}, Test: {test_X.shape}")
        print(f"    Training NSGP (kernel=matern, time=local_per) with float64...")
        
        model = train_nsgp(dataloader, kernel='matern', time_kernel='local_per', iters=50, restarts=2)
        
        if model is None:
            print("    FAILED - skipping")
            continue
        
        pred_y, pred_var = predict_nsgp(model, test_X, dataloader.y_mean)
        y_true = test_y.numpy()
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_true, pred_y))
        mae = mean_absolute_error(y_true, pred_y)
        r2 = r2_score(y_true, pred_y)
        cat_acc, off1, thresh = compute_aqi_metrics(y_true, pred_y)
        
        print(f"    RMSE: {rmse:.2f}, Cat.Acc: {cat_acc*100:.1f}%")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': pred_y,
            'y_var': pred_var
        })
        pred_df.to_csv(f'{results_path}{config_name}_fold{fold}.csv', index=False)
        
        all_results.append({
            'config': config_name,
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'category_accuracy': cat_acc,
            'off_by_one_accuracy': off1,
            'threshold_90_accuracy': thresh
        })
    
    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        df = pd.DataFrame(all_results)
        avg = df.groupby('config').mean(numeric_only=True)
        print(avg)
        
        # Append to main results
        main_results_file = f'{results_path}nsgp_ancl_results.csv'
        if os.path.exists(main_results_file):
            main_df = pd.read_csv(main_results_file)
            # Update A_N_C_L row
            if config_name in main_df['config'].values:
                for col in avg.columns:
                    main_df.loc[main_df['config'] == config_name, col] = avg[col].values[0]
            else:
                new_row = {'config': config_name, 'latex': 'ANCL'}
                for col in avg.columns:
                    new_row[col] = avg[col].values[0]
                main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)
            main_df.to_csv(main_results_file, index=False)
            print(f"\nUpdated {main_results_file}")

if __name__ == '__main__':
    main()
