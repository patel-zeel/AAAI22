"""
Run all baselines from the AAAI22 paper and compute AQI metrics.
Baselines: IDW, KNN, RF, XGB (GradientBoosting), ADAIN, GP (ours)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from polire import IDW
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/time_feature/'
    res_path = os.path.dirname(os.path.abspath(__file__)) + '/experiments/results/'

# Indian National AQI breakpoints for PM2.5 (24-hour average, μg/m³)
INDIA_AQI_BREAKPOINTS = [
    (0, 30, 0, 50, 'Good'),
    (31, 60, 51, 100, 'Satisfactory'),
    (61, 90, 101, 200, 'Moderately Polluted'),
    (91, 120, 201, 300, 'Poor'),
    (121, 250, 301, 400, 'Very Poor'),
    (251, 500, 401, 500, 'Severe'),
]

def pm25_to_aqi_category(pm25, breakpoints=INDIA_AQI_BREAKPOINTS):
    """Convert PM2.5 concentration to AQI category."""
    for c_lo, c_hi, i_lo, i_hi, category in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return category
    if pm25 > 500:
        return breakpoints[-1][4]
    return breakpoints[0][4]

def compute_aqi_metrics(y_true, y_pred, breakpoints=INDIA_AQI_BREAKPOINTS):
    """Compute AQI category-based metrics."""
    true_categories = [pm25_to_aqi_category(y, breakpoints) for y in y_true]
    pred_categories = [pm25_to_aqi_category(y, breakpoints) for y in y_pred]
    
    # Category accuracy
    correct = sum(1 for t, p in zip(true_categories, pred_categories) if t == p)
    category_accuracy = correct / len(true_categories)
    
    # Off-by-one accuracy
    categories = [bp[4] for bp in breakpoints]
    off_by_one_correct = 0
    for t, p in zip(true_categories, pred_categories):
        t_idx = categories.index(t)
        p_idx = categories.index(p)
        if abs(t_idx - p_idx) <= 1:
            off_by_one_correct += 1
    off_by_one_accuracy = off_by_one_correct / len(true_categories)
    
    # Threshold accuracy (90 μg/m³)
    threshold = 90.0
    true_above = np.array(y_true) > threshold
    pred_above = np.array(y_pred) > threshold
    threshold_accuracy = np.mean(true_above == pred_above)
    
    return {
        'category_accuracy': category_accuracy,
        'off_by_one_accuracy': off_by_one_accuracy,
        'threshold_90_accuracy': threshold_accuracy,
    }

class DataLoader:
    """Data loader for Beijing air quality data.
    
    Note: Baselines use 'mar' data (one-hot encoded features, 36 columns)
    while GP uses 'mar_nsgp' data (categorical features, 12 columns).
    Both have the same train/test splits and PM2.5 targets.
    """
    def __init__(self, fold, use_minmax=False, for_gp=False):
        self.Xscaler = MinMaxScaler() if use_minmax else StandardScaler()
        self.fold = fold
        # Baselines use 'mar', GP uses 'mar_nsgp'
        self.suffix = 'mar_nsgp' if for_gp else 'mar'
        self.start = '2015-03-01'
        self.end = '2015-03-31'
        
    def get_feature_cols(self, data):
        """Get feature columns - exclude target and metadata."""
        exclude = ['time', 'station_id', 'PM25_Concentration', 'filled']
        return [c for c in data.columns if c not in exclude]
        
    def common(self, data):
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data[self.start:self.end]
        self.Xcols = self.get_feature_cols(data)
        X = data[self.Xcols]
        y = data[['PM25_Concentration']]
        return X, y, data
    
    def load_train(self):
        train_data = pd.read_csv(Config.data_path + 'fold' + self.fold + '/train_data_' + self.suffix + '.csv.gz')
        X, y, self.train_data = self.common(train_data)
        X = self.Xscaler.fit_transform(X)
        return X, y.values.ravel()
    
    def load_test(self):
        test_data = pd.read_csv(Config.data_path + 'fold' + self.fold + '/test_data_' + self.suffix + '.csv.gz')
        # For 'mar' data, PM2.5 is in test_data itself
        # For 'mar_nsgp', it's in separate test_output file
        if self.suffix == 'mar_nsgp':
            test_output = pd.read_csv(Config.data_path + 'fold' + self.fold + '/test_output_' + self.suffix + '.csv.gz')
            test_data['PM25_Concentration'] = test_output['PM25_Concentration'].values
        
        X, y, self.test_data = self.common(test_data)
        X = self.Xscaler.transform(X)
        return X, y.values.ravel()

def run_idw_baseline(fold):
    """Run IDW baseline using polire library with MinMaxScaler (as in original paper)."""
    dataloader = DataLoader(fold, use_minmax=True, for_gp=False)  # Use 'mar' data
    X_train, y_train = dataloader.load_train()
    X_test, y_test = dataloader.load_test()
    
    # Use polire IDW with exponent=3 as in original notebook
    model = IDW(exponent=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    aqi = compute_aqi_metrics(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'category_accuracy': aqi['category_accuracy'],
        'off_by_one_accuracy': aqi['off_by_one_accuracy'],
        'threshold_90_accuracy': aqi['threshold_90_accuracy'],
        'y_pred': y_pred
    }

def run_baseline(name, X_train, y_train, X_test, y_test):
    """Run a single baseline and return metrics."""
    if name == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif name == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif name == 'XGB':
        # Original paper uses sklearn GradientBoostingRegressor, not XGBoost
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    aqi = compute_aqi_metrics(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'category_accuracy': aqi['category_accuracy'],
        'off_by_one_accuracy': aqi['off_by_one_accuracy'],
        'threshold_90_accuracy': aqi['threshold_90_accuracy'],
        'y_pred': y_pred
    }

def main():
    print("=" * 80)
    print("Baseline Comparison with AQI Metrics")
    print("=" * 80)
    
    baselines = ['KNN', 'RF', 'XGB']  # IDW handled separately
    
    # Store results
    all_results = {name: [] for name in baselines}
    all_results['IDW'] = []  # Add IDW separately
    
    for fold in ['0', '1', '2']:
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print("=" * 80)
        
        # Run IDW separately (uses MinMaxScaler)
        print(f"\n  IDW (polire)...")
        idw_results = run_idw_baseline(fold)
        all_results['IDW'].append(idw_results)
        print(f"    RMSE: {idw_results['rmse']:.2f}, Cat.Acc: {idw_results['category_accuracy']:.1%}, Off-by-1: {idw_results['off_by_one_accuracy']:.1%}")
        
        # Load data for other baselines (StandardScaler, use 'mar' data)
        dataloader = DataLoader(fold, use_minmax=False, for_gp=False)
        X_train, y_train = dataloader.load_train()
        X_test, y_test = dataloader.load_test()
        
        print(f"Train: {X_train.shape[0]} x {X_train.shape[1]}, Test: {X_test.shape[0]}")
        
        for name in baselines:
            print(f"\n  {name}...")
            results = run_baseline(name, X_train, y_train, X_test, y_test)
            all_results[name].append(results)
            print(f"    RMSE: {results['rmse']:.2f}, Cat.Acc: {results['category_accuracy']:.1%}, Off-by-1: {results['off_by_one_accuracy']:.1%}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - Average Across Folds")
    print("=" * 80)
    
    # Also add GP results from previous run
    gp_results = pd.read_csv(Config.res_path + 'gp_results_with_aqi.csv')
    
    print("\n| Model | RMSE | MAE | R² | Cat.Acc | Off-by-1 | Thresh(90) |")
    print("|-------|------|-----|-------|---------|----------|------------|")
    
    summary_data = []
    
    all_baselines = ['IDW'] + baselines  # IDW first
    for name in all_baselines:
        avg_rmse = np.mean([r['rmse'] for r in all_results[name]])
        avg_mae = np.mean([r['mae'] for r in all_results[name]])
        avg_r2 = np.mean([r['r2'] for r in all_results[name]])
        avg_cat = np.mean([r['category_accuracy'] for r in all_results[name]])
        avg_off1 = np.mean([r['off_by_one_accuracy'] for r in all_results[name]])
        avg_thresh = np.mean([r['threshold_90_accuracy'] for r in all_results[name]])
        
        print(f"| {name:5s} | {avg_rmse:.2f} | {avg_mae:.2f} | {avg_r2:.3f} | {avg_cat:.1%} | {avg_off1:.1%} | {avg_thresh:.1%} |")
        
        summary_data.append({
            'Model': name,
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'R2': avg_r2,
            'Category_Accuracy': avg_cat,
            'Off_by_One_Accuracy': avg_off1,
            'Threshold_90_Accuracy': avg_thresh
        })
    
    # Add GP results
    avg_rmse = gp_results['rmse'].mean()
    avg_mae = gp_results['mae'].mean()
    avg_r2 = gp_results['r2'].mean()
    avg_cat = gp_results['category_accuracy'].mean()
    avg_off1 = gp_results['off_by_one_accuracy'].mean()
    avg_thresh = gp_results['threshold_accuracy'].mean()
    
    print(f"| {'GP':5s} | {avg_rmse:.2f} | {avg_mae:.2f} | {avg_r2:.3f} | {avg_cat:.1%} | {avg_off1:.1%} | {avg_thresh:.1%} |")
    
    summary_data.append({
        'Model': 'GP (Ours)',
        'RMSE': avg_rmse,
        'MAE': avg_mae,
        'R2': avg_r2,
        'Category_Accuracy': avg_cat,
        'Off_by_One_Accuracy': avg_off1,
        'Threshold_90_Accuracy': avg_thresh
    })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Config.res_path + 'baseline_comparison.csv', index=False)
    print(f"\nResults saved to {Config.res_path}baseline_comparison.csv")

if __name__ == '__main__':
    main()
