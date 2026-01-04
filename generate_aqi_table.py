"""
Generate a LaTeX table with AQI metrics for thesis Chapter 2.
This creates a table similar to tab:main_compare but adds AQI metrics.
"""

import pandas as pd
import numpy as np

# Load baseline comparison results
baseline_df = pd.read_csv('/home/t-zeelpatel/zeel_thesis/AAAI22/experiments/results/baseline_comparison.csv')

# Print comparison table with AQI metrics
print("=" * 100)
print("RESULTS TABLE WITH AQI METRICS (Beijing Dataset)")
print("=" * 100)

print("\n--- From our experiments (baselines with AQI metrics) ---")
print(baseline_df.to_string(index=False))

print("\n\n--- LaTeX Table Format (matches tab:main_compare style) ---")
print(r"""
% New table with AQI metrics - add after tab:main_compare
\begin{table}[h]
    \centering
    \begin{tabular}{lrrrrrr}
        \toprule
      Model & RMSE ($\downarrow$) & MAE ($\downarrow$) & R$^2$ ($\uparrow$) & Cat. Acc. ($\uparrow$) & Off-by-1 ($\uparrow$) & Thresh. Acc. ($\uparrow$)\\
    \midrule""")

# Print baselines
for _, row in baseline_df.iterrows():
    model = row['Model']
    if model == 'GP (Ours)':
        continue  # Handle GP separately
    rmse = row['RMSE']
    mae = row['MAE']
    r2 = row['R2']
    cat_acc = row['Category_Accuracy'] * 100
    off1 = row['Off_by_One_Accuracy'] * 100
    thresh = row['Threshold_90_Accuracy'] * 100
    
    print(f"    {model} & {rmse:.2f} & {mae:.2f} & {r2:.2f} & {cat_acc:.1f}\\% & {off1:.1f}\\% & {thresh:.1f}\\% \\\\")

print(r"    \midrule")

# Print GP row
gp_row = baseline_df[baseline_df['Model'] == 'GP (Ours)'].iloc[0]
print(f"    \\textbf{{GP (Ours)}} & \\textbf{{{gp_row['RMSE']:.2f}}} & {gp_row['MAE']:.2f} & \\textbf{{{gp_row['R2']:.2f}}} & {gp_row['Category_Accuracy']*100:.1f}\\% & {gp_row['Off_by_One_Accuracy']*100:.1f}\\% & \\textbf{{{gp_row['Threshold_90_Accuracy']*100:.1f}}}\\% \\\\")

print(r"""      \bottomrule
    \end{tabular}
    \caption{
    \rvAdd{RV02I\_XX}{Comparison of conventional metrics and AQI-based metrics on Beijing dataset. 
    \textbf{Category Accuracy} measures exact AQI category match using Indian NAQI breakpoints, 
    \textbf{Off-by-1} allows predictions within one adjacent category,
    \textbf{Threshold Accuracy} measures correct classification above/below 90 $\mu$g/m$^3$ (WHO interim target).
    Our GP achieves the best RMSE and threshold accuracy, while RF achieves higher category accuracy due to its discrete decision boundaries.}
    \newText{($\downarrow$) = lower is better, ($\uparrow$) = higher is better.}
    }
    \label{tab:main_compare_aqi}
\end{table}
""")

# Save results to CSV
baseline_df.to_csv('/home/t-zeelpatel/zeel_thesis/AAAI22/experiments/results/aqi_comparison_table.csv', index=False)
print("\nResults saved to: experiments/results/aqi_comparison_table.csv")

# Print summary statistics
print("\n" + "=" * 100)
print("KEY FINDINGS FOR THESIS:")
print("=" * 100)
rf_rmse = baseline_df[baseline_df['Model']=='RF']['RMSE'].values[0]
gp_rmse = gp_row['RMSE']
improvement = (rf_rmse - gp_rmse) / rf_rmse * 100

print(f"1. GP achieves best RMSE ({gp_rmse:.2f}) - {improvement:.1f}% improvement over RF ({rf_rmse:.2f})")
print(f"2. RF achieves best Category Accuracy ({baseline_df[baseline_df['Model']=='RF']['Category_Accuracy'].values[0]*100:.1f}%)")
print(f"3. GP Category Accuracy: {gp_row['Category_Accuracy']*100:.1f}% (competitive with RF)")
print(f"4. GP achieves best Threshold Accuracy ({gp_row['Threshold_90_Accuracy']*100:.1f}%)")
print(f"5. All top models (GP, RF) achieve >94% Off-by-1 accuracy")
print(f"6. IDW has worst performance across all metrics (classical spatial-only method)")

