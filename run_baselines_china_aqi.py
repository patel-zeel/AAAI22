"""
Run all baselines from the AAAI22 paper and compute AQI metrics using CHINA's standard.
Baselines: IDW, KNN, RF, XGB
"""

import os
import sys
import numpy as np
import pandas as pd
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

# China HJ 633-2012 PM2.5 AQI breakpoints (24-hour average, μg/m³)
CHINA_AQI_BREAKPOINTS = [
    (0, 35, 'Excellent'),
    (36, 75, 'Good'),
    (76, 115, 'Lightly Polluted'),
    (116, 150, 'Moderately Polluted'),
    (151, 250, 'Heavily Polluted'),
    (251, 500, 'Severely Polluted'),
]
CATEGORIES = [bp[2] for bp in CHINA_AQI_BREAKPOINTS]

def pm25_to_aqi_category(pm25):
    """Convert PM2.5 concentration to AQI category using China standard."""
    for lo, hi, category in CHINA_AQI_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return category
    if pm25 > 500:
        return 'Severely Polluted'
    return 'Excellent'

def compute_category_wise_accuracy(y_true, y_pred):
    """Compute overall and per-category accuracy."""
    true_cats = [pm25_to_aqi_category(y) for y in y_true]
    pred_cats = [pm25_to_aqi_category(y) for y in y_pred]
    
    overall_acc = sum(1 for t, p in zip(true_cats, pred_cats) if t == p) / len(true_cats)
    
    cat_accs = {}
    cat_counts = {}
    for cat in CATEGORIES:
        indices = [i for i, t in enumerate(true_cats) if t == cat]
        if len(indices) > 0:
            correct = sum(1 for i in indices if true_cats[i] == pred_cats[i])
            cat_accs[cat] = correct / len(indices)
            cat_counts[cat] = len(indices)
        else:
            cat_accs[cat] = np.nan
            cat_counts[cat] = 0
    
    return overall_acc, cat_accs, cat_counts

class DataLoader:
    """Data loader for Beijing air quality data."""
    def __init__(self, fold, use_minmax=False):
        self.Xscaler = MinMaxScaler() if use_minmax else StandardScaler()
        self.fold = fold
        self.suffix = 'mar'  # Baselines use 'mar' data
        self.start = '2015-03-01'
        self.end = '2015-03-31'
        
    def get_feature_cols(self, data):
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
        X, y, self.test_data = self.common(test_data)
        X = self.Xscaler.transform(X)
        return X, y.values.ravel()

def run_idw(fold):
    """Run IDW baseline."""
    dataloader = DataLoader(fold, use_minmax=True)
    X_train, y_train = dataloader.load_train()
    X_test, y_test = dataloader.load_test()
    
    model = IDW(exponent=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

def run_knn(fold):
    """Run KNN baseline."""
    dataloader = DataLoader(fold, use_minmax=False)
    X_train, y_train = dataloader.load_train()
    X_test, y_test = dataloader.load_test()
    
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

def run_rf(fold):
    """Run Random Forest baseline."""
    dataloader = DataLoader(fold, use_minmax=False)
    X_train, y_train = dataloader.load_train()
    X_test, y_test = dataloader.load_test()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

def run_xgb(fold):
    """Run XGBoost (GradientBoosting) baseline."""
    dataloader = DataLoader(fold, use_minmax=False)
    X_train, y_train = dataloader.load_train()
    X_test, y_test = dataloader.load_test()
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

def main():
    print("=" * 100)
    print("BASELINE COMPARISON WITH CHINA AQI CATEGORIES (HJ 633-2012)")
    print("=" * 100)
    
    baselines = {
        'IDW': run_idw,
        'KNN': run_knn,
        'RF': run_rf,
        'XGB': run_xgb,
    }
    
    results = {name: {'y_true': [], 'y_pred': []} for name in baselines}
    
    # Run all baselines across all folds
    for fold in ['0', '1', '2']:
        print(f"\nFold {fold}:")
        for name, run_fn in baselines.items():
            y_test, y_pred = run_fn(fold)
            results[name]['y_true'].extend(y_test)
            results[name]['y_pred'].extend(y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"  {name}: RMSE={rmse:.2f}")
    
    # Compute final metrics
    print("\n" + "=" * 100)
    print("CATEGORY-WISE ACCURACY (CHINA AQI)")
    print("=" * 100)
    
    # Header
    print(f"\n{'Model':<8} | {'Overall':>8} | {'Excellent':>10} | {'Good':>10} | {'Lightly P.':>10} | {'Mod. P.':>10} | {'Heavily P.':>10} | {'Severely P.':>10}")
    print("-" * 100)
    
    final_results = []
    for name in baselines:
        y_true = np.array(results[name]['y_true'])
        y_pred = np.array(results[name]['y_pred'])
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_acc, cat_accs, cat_counts = compute_category_wise_accuracy(y_true, y_pred)
        
        print(f"{name:<8} | {overall_acc*100:>7.1f}% | {cat_accs['Excellent']*100:>9.1f}% | {cat_accs['Good']*100:>9.1f}% | {cat_accs['Lightly Polluted']*100:>9.1f}% | {cat_accs['Moderately Polluted']*100:>9.1f}% | {cat_accs['Heavily Polluted']*100:>9.1f}% | {cat_accs['Severely Polluted']*100:>9.1f}%")
        
        final_results.append({
            'Model': name,
            'RMSE': rmse,
            'Overall': overall_acc,
            'Excellent': cat_accs['Excellent'],
            'Good': cat_accs['Good'],
            'Lightly_Polluted': cat_accs['Lightly Polluted'],
            'Moderately_Polluted': cat_accs['Moderately Polluted'],
            'Heavily_Polluted': cat_accs['Heavily Polluted'],
            'Severely_Polluted': cat_accs['Severely Polluted'],
        })
    
    # Save results
    df = pd.DataFrame(final_results)
    df.to_csv(Config.res_path + 'baseline_china_aqi_results.csv', index=False)
    print(f"\nSaved to {Config.res_path}baseline_china_aqi_results.csv")
    
    # Print LaTeX table format
    print("\n" + "=" * 100)
    print("LATEX TABLE FORMAT:")
    print("=" * 100)
    for r in final_results:
        print(f"        {r['Model']} & {r['Overall']*100:.1f}\\% & {r['Excellent']*100:.1f}\\% & {r['Good']*100:.1f}\\% & {r['Lightly_Polluted']*100:.1f}\\% & {r['Moderately_Polluted']*100:.1f}\\% & {r['Heavily_Polluted']*100:.1f}\\% & {r['Severely_Polluted']*100:.1f}\\% \\\\")

if __name__ == '__main__':
    main()
