"""
Simple script to run GP experiment on Beijing air quality data and compute AQI metrics.
This avoids the SLURM cluster dependency.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent path for nsgp imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
class Config:
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/time_feature/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res_path = os.path.dirname(os.path.abspath(__file__)) + '/experiments/results/'

# AQI breakpoints for PM2.5 (US EPA standard, in μg/m³)
AQI_BREAKPOINTS = [
    (0, 12.0, 0, 50, 'Good'),
    (12.1, 35.4, 51, 100, 'Moderate'),
    (35.5, 55.4, 101, 150, 'Unhealthy for Sensitive Groups'),
    (55.5, 150.4, 151, 200, 'Unhealthy'),
    (150.5, 250.4, 201, 300, 'Very Unhealthy'),
    (250.5, 500.4, 301, 500, 'Hazardous'),
]

# Indian National AQI breakpoints for PM2.5 (24-hour average, μg/m³)
# Source: CPCB India
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
        return breakpoints[-1][4]  # Worst category
    return breakpoints[0][4]  # Best category

def compute_aqi_metrics(y_true, y_pred, breakpoints=INDIA_AQI_BREAKPOINTS):
    """Compute AQI category-based metrics."""
    true_categories = [pm25_to_aqi_category(y, breakpoints) for y in y_true]
    pred_categories = [pm25_to_aqi_category(y, breakpoints) for y in y_pred]
    
    # Category accuracy
    correct = sum(1 for t, p in zip(true_categories, pred_categories) if t == p)
    category_accuracy = correct / len(true_categories)
    
    # Off-by-one accuracy (correct or within one category)
    categories = [bp[4] for bp in breakpoints]
    off_by_one_correct = 0
    for t, p in zip(true_categories, pred_categories):
        t_idx = categories.index(t)
        p_idx = categories.index(p)
        if abs(t_idx - p_idx) <= 1:
            off_by_one_correct += 1
    off_by_one_accuracy = off_by_one_correct / len(true_categories)
    
    # Threshold crossing accuracy (90 μg/m³ is "Moderately Polluted" to "Poor" in India)
    threshold = 90.0
    true_above = np.array(y_true) > threshold
    pred_above = np.array(y_pred) > threshold
    threshold_accuracy = np.mean(true_above == pred_above)
    
    # Per-category precision/recall
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

class DataLoader:
    """Data loader for Beijing air quality data."""
    def __init__(self, fold, Xcols):
        self.Xscaler = StandardScaler()
        self.fold = fold
        self.Xcols = Xcols.split('@')
        self.suffix = 'mar_nsgp'
        self.start = '2015-03-01'
        self.end = '2015-03-31'
        self.all_cont_cols = ['longitude', 'latitude', 'temperature', 'humidity', 'wind_speed', 'delta_t']
        
    def common(self, data):
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data[self.start:self.end]
        X = data[self.Xcols]
        y = data[['PM25_Concentration']]
        return X, y, data
    
    def load_train(self):
        train_data = pd.read_csv(Config.data_path + 'fold' + self.fold + '/train_data_' + self.suffix + '.csv.gz')
        X, y, self.train_data = self.common(train_data)
        
        self.cont_cols = [i for i in self.Xcols if i in self.all_cont_cols]
        X[self.cont_cols] = self.Xscaler.fit_transform(X[self.cont_cols])
        
        self.cat_indicator = [0 if i in self.cont_cols else 1 for i in X.columns]
        self.y_mean = y.values.mean()
        y = y - self.y_mean
        
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        return X, y
    
    def load_test(self):
        test_data = pd.read_csv(Config.data_path + 'fold' + self.fold + '/test_data_' + self.suffix + '.csv.gz')
        test_output = pd.read_csv(Config.data_path + 'fold' + self.fold + '/test_output_' + self.suffix + '.csv.gz')
        test_data['PM25_Concentration'] = test_output['PM25_Concentration'].values
        
        X, y, self.test_data = self.common(test_data)
        X[self.cont_cols] = self.Xscaler.transform(X[self.cont_cols])
        
        # Center y using training mean (same as training)
        y = y - self.y_mean
        
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        return X, y

class ExactGPModel(gpytorch.models.ExactGP):
    """GP model with ARD RBF kernel."""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(X_train, y_train, n_iter=100, device='cpu'):
    """Train a GP model."""
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(X_train, y_train.ravel(), likelihood).to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train.ravel())
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f'  Iter {i+1}/{n_iter} - Loss: {loss.item():.3f}')
    
    return model, likelihood

def predict_gp(model, likelihood, X_test, y_mean, device='cpu'):
    """Make predictions with trained GP."""
    model.eval()
    likelihood.eval()
    
    X_test = X_test.to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_test))
        pred_mean = pred.mean.cpu().numpy() + y_mean
        pred_var = pred.variance.cpu().numpy()
    
    return pred_mean, pred_var

def main():
    print("=" * 70)
    print("GP Experiment on Beijing Air Quality Data with AQI Metrics")
    print("=" * 70)
    print(f"Device: {Config.device}")
    
    # Features to use
    Xcols = '@'.join(['longitude', 'latitude', 'humidity', 'temperature', 'wind_speed', 'delta_t'])
    
    all_results = []
    all_y_true = []
    all_y_pred = []
    
    for fold in ['0', '1', '2']:
        print(f"\n{'='*70}")
        print(f"FOLD {fold}")
        print("=" * 70)
        
        # Load data
        dataloader = DataLoader(fold, Xcols)
        X_train, y_train = dataloader.load_train()
        X_test, y_test = dataloader.load_test()
        
        print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # Train GP (use full training data with GPU)
        device = Config.device
        model, likelihood = train_gp(X_train, y_train, n_iter=100, device=device)
        
        # Predict
        pred_mean, pred_var = predict_gp(model, likelihood, X_test, dataloader.y_mean, device=device)
        y_true = y_test.numpy().ravel() + dataloader.y_mean
        
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
        
        # AQI metrics (China standard)
        aqi_metrics = compute_aqi_metrics(y_true, pred_mean, INDIA_AQI_BREAKPOINTS)
        
        print(f"\n  AQI-Based Metrics (China Standard):")
        print(f"    Exact Category Accuracy:     {aqi_metrics['category_accuracy']:.1%}")
        print(f"    Off-by-One Category Accuracy:{aqi_metrics['off_by_one_accuracy']:.1%}")
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
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY (Across All Folds)")
    print("=" * 70)
    
    avg_rmse = np.mean([r['rmse'] for r in all_results])
    avg_mae = np.mean([r['mae'] for r in all_results])
    avg_r2 = np.mean([r['r2'] for r in all_results])
    
    # Compute overall AQI metrics
    overall_aqi = compute_aqi_metrics(all_y_true, all_y_pred, INDIA_AQI_BREAKPOINTS)
    
    print(f"\nAverage Standard Metrics:")
    print(f"  RMSE: {avg_rmse:.2f} μg/m³")
    print(f"  MAE:  {avg_mae:.2f} μg/m³")
    print(f"  R²:   {avg_r2:.3f}")
    
    print(f"\nOverall AQI-Based Metrics (China Standard):")
    print(f"  Exact Category Accuracy:      {overall_aqi['category_accuracy']:.1%}")
    print(f"  Off-by-One Category Accuracy: {overall_aqi['off_by_one_accuracy']:.1%}")
    print(f"  Threshold (75 μg/m³) Accuracy: {overall_aqi['threshold_75_accuracy']:.1%}")
    
    print(f"\nOverall Per-Category Performance:")
    for cat, metrics in overall_aqi['category_metrics'].items():
        if metrics['count'] > 0:
            print(f"  {cat:20s}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1']:.2f} (n={metrics['count']})")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Config.res_path + 'gp_results_with_aqi.csv', index=False)
    print(f"\nResults saved to {Config.res_path}gp_results_with_aqi.csv")
    
    # Also save detailed predictions
    predictions_df = pd.DataFrame({
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'true_category': [pm25_to_aqi_category(y) for y in all_y_true],
        'pred_category': [pm25_to_aqi_category(y) for y in all_y_pred]
    })
    predictions_df.to_csv(Config.res_path + 'gp_predictions.csv', index=False)
    print(f"Predictions saved to {Config.res_path}gp_predictions.csv")

if __name__ == '__main__':
    main()
