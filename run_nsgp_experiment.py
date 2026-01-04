"""
Run the full NSGP model from the AAAI22 paper and compute AQI metrics.
This script directly calls the training and testing pipeline.
"""

import os
import sys

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'nonstat_gp_cat'))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import from the repo
from config import Config
from data_loader import Data
from nsgp_sgd import NSGP
from nsgp.utils.inducing_functions import f_kmeans

# Indian National AQI breakpoints for PM2.5 (24-hour average, μg/m³)
# Source: CPCB India
INDIA_AQI_BREAKPOINTS = [
    (0, 30, 'Good'),
    (31, 60, 'Satisfactory'),
    (61, 90, 'Moderately Polluted'),
    (91, 120, 'Poor'),
    (121, 250, 'Very Poor'),
    (251, 500, 'Severe'),
]

def pm25_to_aqi_category(pm25):
    """Convert PM2.5 concentration to Indian AQI category."""
    for c_lo, c_hi, category in INDIA_AQI_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return category
    if pm25 > 500:
        return 'Severe'
    return 'Good'

def compute_aqi_metrics(y_true, y_pred):
    """Compute AQI category-based metrics."""
    true_categories = [pm25_to_aqi_category(y) for y in y_true]
    pred_categories = [pm25_to_aqi_category(y) for y in y_pred]
    
    # Category accuracy
    correct = sum(1 for t, p in zip(true_categories, pred_categories) if t == p)
    category_accuracy = correct / len(true_categories)
    
    # Off-by-one accuracy
    categories = [bp[2] for bp in INDIA_AQI_BREAKPOINTS]
    off_by_one_correct = 0
    for t, p in zip(true_categories, pred_categories):
        t_idx = categories.index(t)
        p_idx = categories.index(p)
        if abs(t_idx - p_idx) <= 1:
            off_by_one_correct += 1
    off_by_one_accuracy = off_by_one_correct / len(true_categories)
    
    # Threshold accuracy (90 μg/m³ = Moderately Polluted to Poor boundary)
    threshold = 90.0
    true_above = np.array(y_true) > threshold
    pred_above = np.array(y_pred) > threshold
    threshold_accuracy = np.mean(true_above == pred_above)
    
    # Per-category metrics
    category_metrics = {}
    for cat in categories:
        true_cat = [t == cat for t in true_categories]
        pred_cat = [p == cat for p in pred_categories]
        
        tp = sum(1 for t, p in zip(true_cat, pred_cat) if t and p)
        fp = sum(1 for t, p in zip(true_cat, pred_cat) if not t and p)
        fn = sum(1 for t, p in zip(true_cat, pred_cat) if t and not p)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics[cat] = {'precision': precision, 'recall': recall, 'f1': f1, 'count': sum(true_cat)}
    
    return {
        'category_accuracy': category_accuracy,
        'off_by_one_accuracy': off_by_one_accuracy,
        'threshold_75_accuracy': threshold_accuracy,
        'category_metrics': category_metrics
    }

def train_nsgp(fold, Xcols, kernel='rbf', time_kernel='loc_periodic', n_iters=40, n_restarts=3, batch_div=4):
    """Train NSGP model on a fold."""
    print(f"  Loading data for fold {fold}...")
    dataloader = Data(fold, Xcols=Xcols)
    X, y, _ = dataloader.load_train()
    
    # Prepare inducing points
    X_bar = []
    Num_Ind = 100
    for dim in range(X.shape[1]):
        if X[:, dim].unique().shape[0] < Num_Ind:
            X_bar.append(X[:, dim].unique().reshape(-1, 1).to(Config.device))
        else:
            X_bar.append(f_kmeans(X[:, dim:dim+1], num_inducing_points=Num_Ind, random_state=0).to(Config.device))
    
    print(f"  X shape: {X.shape}, Inducing points created")
    
    # Training setup
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    train_dataset = TensorDataset(X, y)
    torch.manual_seed(0)
    trainloader = TorchDataLoader(train_dataset, batch_size=X.shape[0]//batch_div, shuffle=True)
    
    ystd = torch.sqrt(y.var())
    best_loss = np.inf
    best_model = None
    
    def init_vars(model):
        with torch.no_grad():
            if time_kernel in ['loc_periodic', 'periodic']:
                torch.nn.init.uniform_(model.period, a=0.03, b=0.07)
            torch.nn.init.uniform_(model.local_gp_ls, a=0.5, b=3)
            torch.nn.init.uniform_(model.local_gp_std, a=0.5, b=3)
            torch.nn.init.uniform_(model.local_gp_noise_std, a=0.5, b=3)
            for param in model.local_ls:
                torch.nn.init.uniform_(param, a=0.5, b=3)
            torch.nn.init.uniform_(model.global_gp_std, a=0.5, b=3)
            torch.nn.init.uniform_(model.global_gp_noise_std, a=0.5, b=3)
    
    for restart in range(n_restarts):
        print(f"  Restart {restart+1}/{n_restarts}...")
        torch.manual_seed(restart)
        model = NSGP(X_bar, dataloader.cat_indicator, dataloader.time_indicator,
                     random_state=restart, jitter=10**-3, use_Trans=False, m=3,
                     kernel=kernel, timekernel=time_kernel).to(Config.device)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        init_vars(model)
        
        for i in range(n_iters):
            loss_val = 0
            for bi, (X_batch, y_batch) in enumerate(trainloader):
                optim.zero_grad()
                try:
                    loss = model(X_batch.to(Config.device), y_batch.to(Config.device))
                    loss_val += loss.item()
                    loss.backward()
                    optim.step()
                    with torch.no_grad():
                        for param in model.parameters():
                            torch.clamp_(param, min=10**-5)
                except Exception as e:
                    print(f"    Iter {i}, Batch {bi} failed: {e}")
                    loss_val = np.inf
                    break
            
            final_loss = loss_val / len(trainloader)
            if (i + 1) % 10 == 0:
                print(f"    Iter {i+1}/{n_iters}, Loss: {final_loss:.4f}")
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            print(f"  New best loss: {best_loss:.4f}")
    
    return best_model, dataloader

def predict_nsgp(model, dataloader):
    """Make predictions with trained NSGP model."""
    # Load training data (needed for prediction)
    X_train, y_train, _ = dataloader.load_train()
    X_train = X_train.to(Config.device)
    y_train = y_train.to(Config.device)
    
    # Load test data
    X_test, y_test = dataloader.load_test()
    X_test = X_test.to(Config.device)
    
    model.eval()
    with torch.no_grad():
        pred_mean, pred_var = model.predict(X_train, y_train, X_test)
    
    pred_mean = pred_mean.cpu().numpy().ravel() + dataloader.y_mean
    y_true = y_test.numpy().ravel() + dataloader.y_mean
    
    return y_true, pred_mean

def main():
    print("=" * 70)
    print("NSGP Experiment on Beijing Air Quality Data with AQI Metrics")
    print("=" * 70)
    print(f"Device: {Config.device}")
    
    # Features to use (with categorical)
    Xcols = '@'.join(['longitude', 'latitude', 'humidity', 'temperature', 'weather', 'wind_direction', 'wind_speed', 'delta_t'])
    
    all_results = []
    all_y_true = []
    all_y_pred = []
    
    for fold in ['0', '1', '2']:
        print(f"\n{'='*70}")
        print(f"FOLD {fold}")
        print("=" * 70)
        
        # Train NSGP
        model, dataloader = train_nsgp(fold, Xcols, n_iters=40, n_restarts=2, batch_div=4)
        
        # Predict
        y_true, pred_mean = predict_nsgp(model, dataloader)
        
        all_y_true.extend(y_true)
        all_y_pred.extend(pred_mean)
        
        # Standard metrics
        rmse = np.sqrt(mean_squared_error(y_true, pred_mean))
        mae = mean_absolute_error(y_true, pred_mean)
        r2 = r2_score(y_true, pred_mean)
        
        print(f"\n  Standard Metrics:")
        print(f"    RMSE: {rmse:.2f} μg/m³")
        print(f"    MAE:  {mae:.2f} μg/m³")
        print(f"    R²:   {r2:.3f}")
        
        # AQI metrics
        aqi_metrics = compute_aqi_metrics(y_true, pred_mean)
        
        print(f"\n  AQI-Based Metrics (China Standard):")
        print(f"    Exact Category Accuracy:      {aqi_metrics['category_accuracy']:.1%}")
        print(f"    Off-by-One Category Accuracy: {aqi_metrics['off_by_one_accuracy']:.1%}")
        print(f"    Threshold (75 μg/m³) Accuracy: {aqi_metrics['threshold_75_accuracy']:.1%}")
        
        print(f"\n  Per-Category Performance:")
        for cat, metrics in aqi_metrics['category_metrics'].items():
            if metrics['count'] > 0:
                print(f"    {cat:20s}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f} (n={metrics['count']})")
        
        all_results.append({
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'category_accuracy': aqi_metrics['category_accuracy'],
            'off_by_one_accuracy': aqi_metrics['off_by_one_accuracy'],
            'threshold_accuracy': aqi_metrics['threshold_75_accuracy']
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    avg_rmse = np.mean([r['rmse'] for r in all_results])
    overall_aqi = compute_aqi_metrics(all_y_true, all_y_pred)
    
    print(f"\nAverage RMSE: {avg_rmse:.2f} μg/m³")
    print(f"\nOverall AQI Metrics:")
    print(f"  Exact Category Accuracy:      {overall_aqi['category_accuracy']:.1%}")
    print(f"  Off-by-One Category Accuracy: {overall_aqi['off_by_one_accuracy']:.1%}")
    print(f"  Threshold (75 μg/m³) Accuracy: {overall_aqi['threshold_75_accuracy']:.1%}")
    
    # Save results
    os.makedirs(Config.res_path, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Config.res_path + 'nsgp_results_with_aqi.csv', index=False)
    print(f"\nResults saved to {Config.res_path}nsgp_results_with_aqi.csv")

if __name__ == '__main__':
    main()
