"""
Run NSGP experiments for all ANCL configurations and compute AQI metrics.
Configurations from the paper:
- ĀNCL̄: ARD=off, NonStat=off, Cat=off, LocPer=off (baseline GP)
- ANCL̄: ARD=on, NonStat=off, Cat=off, LocPer=off
- ANC̄L̄: ARD=on, NonStat=off, Cat=on, LocPer=off
- AN̄CL: ARD=on, NonStat=off, Cat=on, LocPer=on (best by NLML)
- ANCL̄: ARD=on, NonStat=on, Cat=on, LocPer=off
- ANCL: ARD=on, NonStat=on, Cat=on, LocPer=on
"""

import os
import sys
sys.path.insert(0, '/home/t-zeelpatel/zeel_thesis/AAAI22/nonstat_gp_cat')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import Config

# AQI computation functions
INDIA_AQI_BREAKPOINTS = [
    (0, 30, 0, 50, 'Good'),
    (31, 60, 51, 100, 'Satisfactory'),
    (61, 90, 101, 200, 'Moderate'),
    (91, 120, 201, 300, 'Poor'),
    (121, 250, 301, 400, 'Very Poor'),
    (251, 500, 401, 500, 'Severe'),
]

def pm25_to_aqi_category(pm25, breakpoints=INDIA_AQI_BREAKPOINTS):
    for c_lo, c_hi, i_lo, i_hi, category in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return category
    if pm25 > 500:
        return breakpoints[-1][4]
    return breakpoints[0][4]

def compute_aqi_metrics(y_true, y_pred, breakpoints=INDIA_AQI_BREAKPOINTS):
    true_categories = [pm25_to_aqi_category(y, breakpoints) for y in y_true]
    pred_categories = [pm25_to_aqi_category(y, breakpoints) for y in y_pred]
    
    correct = sum(1 for t, p in zip(true_categories, pred_categories) if t == p)
    category_accuracy = correct / len(true_categories)
    
    categories = [bp[4] for bp in breakpoints]
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

# Configuration mapping from paper notation to experiment params
# Format: (ARD, NonStat, Cat, LocPer) -> (Xcols, kernel, time_kernel)
# ARD is implicit (always use all features)
# NonStat: 'rbf' vs 'nsgp' model
# Cat: categorical columns vs one-hot encoded
# LocPer: 'rbf' vs 'local_per' time kernel

CONFIGS = {
    # ĀNCL̄: baseline stationary GP with one-hot features
    'A_bar_N_bar_C_bar_L_bar': {
        'model': 'gp',  # stationary GP
        'Xcols': 'one_hot',  # one-hot encoded 
        'kernel': 'rbf',
        'time_kernel': 'rbf',
        'latex': r'$\mathbf{\bar{A}\bar{N}\bar{C}\bar{L}}$'
    },
    # ANCL̄: ARD + stationary GP with one-hot features
    'A_N_bar_C_bar_L_bar': {
        'model': 'gp',
        'Xcols': 'one_hot',
        'kernel': 'rbf',
        'time_kernel': 'rbf',
        'latex': r'$\mathbf{A\bar{N}\bar{C}\bar{L}}$'
    },
    # ANC̄L̄: ARD + stationary GP + categorical kernel (no local periodic)
    'A_N_bar_C_L_bar': {
        'model': 'nsgp',  # non-stationary to use cat kernel
        'Xcols': 'cat',
        'kernel': 'rbf',
        'time_kernel': 'rbf',
        'latex': r'$\mathbf{A\bar{N}C\bar{L}}$'
    },
    # AN̄CL: ARD + stationary + categorical + local periodic (best by NLML)
    'A_N_bar_C_L': {
        'model': 'nsgp',
        'Xcols': 'cat',
        'kernel': 'rbf',
        'time_kernel': 'local_per',
        'latex': r'$\mathbf{A\bar{N}CL}$'
    },
    # ANCL̄: all extensions except local periodic
    'A_N_C_L_bar': {
        'model': 'nsgp',
        'Xcols': 'cat',
        'kernel': 'matern',  # non-stationary
        'time_kernel': 'rbf',
        'latex': r'$\mathbf{ANC\bar{L}}$'
    },
    # ANCL: all extensions enabled
    'A_N_C_L': {
        'model': 'nsgp',
        'Xcols': 'cat',
        'kernel': 'matern',
        'time_kernel': 'local_per',
        'latex': r'$\mathbf{ANCL}$'
    },
}

def find_result_file(config_name, fold):
    """Find the result CSV file for a configuration."""
    # Check experiments/results for saved predictions
    res_path = Config.res_path
    
    # Try different naming patterns
    patterns = [
        f"nsgp_{fold}_*.csv",
        f"gp_{fold}_*.csv",
    ]
    
    import glob
    for pattern in patterns:
        files = glob.glob(os.path.join(res_path, pattern))
        if files:
            return files[0]
    
    return None

def main():
    print("=" * 80)
    print("NSGP Variant Experiments with AQI Metrics")
    print("=" * 80)
    
    # Check what result files exist
    res_path = Config.res_path
    print(f"\nChecking results in: {res_path}")
    
    import glob
    csv_files = glob.glob(os.path.join(res_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV result files")
    for f in csv_files[:10]:
        print(f"  - {os.path.basename(f)}")
    
    # Check for model files
    model_files = glob.glob(os.path.join(res_path, "*.model"))
    print(f"\nFound {len(model_files)} model files")
    for f in model_files[:10]:
        print(f"  - {os.path.basename(f)}")
    
    # If no results exist, we need to run experiments
    if len(csv_files) == 0:
        print("\n" + "=" * 80)
        print("NO SAVED RESULTS FOUND - NEED TO RUN EXPERIMENTS")
        print("=" * 80)
        print("\nTo run experiments, use runner.py or run commands like:")
        print("  python gp_train.py nsgp ad 0 40 50 5 4 uni cat rbf local_per")
        print("  python gp_test.py nsgp 0 uni cat rbf local_per")
        return
    
    # Process existing results
    print("\n" + "=" * 80)
    print("Processing existing results...")
    print("=" * 80)
    
    all_results = []
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        if 'gp_predictions' in basename or 'baseline' in basename or 'aqi' in basename:
            continue  # Skip our generated files
            
        print(f"\nProcessing: {basename}")
        try:
            df = pd.read_csv(csv_file)
            if 'pred_mean' not in df.columns or 'PM25_Concentration' not in df.columns:
                print(f"  Missing required columns, skipping")
                continue
            
            y_true = df['PM25_Concentration'].values
            y_pred = df['pred_mean'].values
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            aqi = compute_aqi_metrics(y_true, y_pred)
            
            print(f"  RMSE: {rmse:.2f}, Cat.Acc: {aqi['category_accuracy']*100:.1f}%")
            
            all_results.append({
                'file': basename,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'category_accuracy': aqi['category_accuracy'],
                'off_by_one_accuracy': aqi['off_by_one_accuracy'],
                'threshold_90_accuracy': aqi['threshold_90_accuracy'],
            })
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(res_path, 'nsgp_aqi_results.csv'), index=False)
        print(f"\nSaved results to: {res_path}/nsgp_aqi_results.csv")
        print("\nSummary:")
        print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
