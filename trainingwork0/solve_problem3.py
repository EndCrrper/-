# -*- coding: utf-8 -*-
"""
Problem 3: RTD-GBDT Hybrid Prediction Model
============================================
RTD (Residence Time Distribution) modelling with tau=4h, N=3 CSTR
Convolve FILT_NTU with RTD weights
GBDT model with RTD features + time features + lag features
Predict CW_NTU for 2026-02-01, 02-10, 02-20 hours 7-19
OAT sensitivity analysis
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import math
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 1. Setup: Chinese fonts and output paths
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Register Chinese font
font_path = 'C:/Windows/Fonts/msyh.ttc'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[Setup] Chinese font loaded: {font_name}")
else:
    plt.rcParams['font.family'] = 'sans-serif'
    print("[Setup] msyh.ttc not found, using default sans-serif")

print("=" * 70)
print("Problem 3: RTD-GBDT Hybrid Prediction Model for CW_NTU")
print("=" * 70)

# ============================================================
# 2. Load data
# ============================================================
DATA_PATH = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')
print(f"\n[Data] Loading cleaned data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print(f"[Data] Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"[Data] Date range: {df['DATE'].iloc[0]} to {df['DATE'].iloc[-1]}")

# Parse datetime
df['DATETIME'] = pd.to_datetime(df['DATETIME'], errors='coerce')
df.dropna(subset=['DATETIME'], inplace=True)
df.sort_values('DATETIME', inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# 3. RTD (Residence Time Distribution) Calculation
# ============================================================
print("\n" + "=" * 70)
print("RTD (Residence Time Distribution) Modelling")
print("  Tanks-in-Series: N = 3 CSTR")
print("  Mean residence time: tau = 4.0 hours (2 steps)")
print("=" * 70)

# Parameters
N_CSTR = 3          # Number of CSTRs in series
TAU_HOURS = 4.0      # Mean residence time (hours)
DT = 2.0             # Time step (hours) - data is every 2 hours
TAU_STEPS = int(TAU_HOURS / DT)  # tau in steps

print(f"[RTD] N_CSTR = {N_CSTR}")
print(f"[RTD] tau = {TAU_HOURS}h ({TAU_STEPS} steps)")
print(f"[RTD] dt = {DT}h")

# Calculate RTD weights for a finite window
# For N CSTR in series: E(t) = (t^(N-1) * exp(-t/tau)) / (tau^N * (N-1)!)
# In discrete time steps

MAX_WINDOW = 48  # Maximum RTD window (96 hours)
rtd_weights = np.zeros(MAX_WINDOW)
time_points = np.arange(0, MAX_WINDOW) * DT  # Hours

for k in range(MAX_WINDOW):
    t = time_points[k]
    if t < 0:
        rtd_weights[k] = 0
    else:
        # E(t) = (t^(N-1) * exp(-t/tau)) / (tau^N * (N-1)!)
        numerator = (t ** (N_CSTR - 1)) * np.exp(-t / TAU_HOURS)
        denominator = (TAU_HOURS ** N_CSTR) * math.factorial(N_CSTR - 1)
        rtd_weights[k] = numerator / denominator

# Normalize weights
rtd_weights = rtd_weights / np.sum(rtd_weights)

# Truncate to where weights are significant (> 1% of max)
significant_idx = np.where(rtd_weights > 0.01 * rtd_weights.max())[0]
if len(significant_idx) > 0:
    rtd_window = significant_idx[-1] + 1
else:
    rtd_window = TAU_STEPS * 3
rtd_weights = rtd_weights[:rtd_window]
rtd_weights = rtd_weights / np.sum(rtd_weights)  # Re-normalize

print(f"[RTD] Significant window: {rtd_window} steps ({rtd_window * DT}h)")
print(f"[RTD] RTD weights shape: {rtd_weights.shape}")
print(f"[RTD] First 10 weights: {np.round(rtd_weights[:10], 6)}")

# --- Chart 1: RTD Curve ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RTD curve
time_hrs = np.arange(len(rtd_weights)) * DT
axes[0].bar(time_hrs, rtd_weights, width=DT * 0.8, color='steelblue', alpha=0.7, edgecolor='navy')
axes[0].set_xlabel('Time (hours)')
axes[0].set_ylabel('E(t) - RTD Weight')
axes[0].set_title(f'Residence Time Distribution (N={N_CSTR}, tau={TAU_HOURS}h)')
axes[0].grid(True, alpha=0.3)

# Cumulative RTD (F-curve)
cumulative_rtd = np.cumsum(rtd_weights)
axes[1].plot(time_hrs, cumulative_rtd, 'b-', linewidth=2)
axes[1].axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
axes[1].axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Time (hours)')
axes[1].set_ylabel('F(t) - Cumulative Distribution')
axes[1].set_title('Cumulative RTD (F-curve)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
rtd_curve_path = os.path.join(OUTPUT_DIR, 'p3_rtd_curve.png')
plt.savefig(rtd_curve_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Chart] Saved: {rtd_curve_path}")

# ============================================================
# 4. Apply RTD Convolution to FILT_NTU
# ============================================================
print("\n" + "=" * 70)
print("RTD Convolution: Applying RTD weights to FILT_NTU")
print("=" * 70)

def apply_rtd_convolution(series, weights):
    """Apply RTD convolution to a time series."""
    n = len(series)
    w = len(weights)
    convolved = np.full(n, np.nan)
    for i in range(n):
        if i >= w - 1:
            segment = series.iloc[i - w + 1 : i + 1][::-1].values  # Reverse for convolution
            if len(segment) == w and not np.any(np.isnan(segment)):
                convolved[i] = np.sum(segment * weights)
    return convolved

# Apply RTD convolution to FILT_NTU
if 'FILT_NTU' in df.columns:
    df['FILT_NTU_RTD'] = apply_rtd_convolution(df['FILT_NTU'], rtd_weights)
    print(f"[RTD] RTD-convolved FILT_NTU created.")

    # Also apply to RW_NTU for additional RTD features
    if 'RW_NTU' in df.columns:
        df['RW_NTU_RTD'] = apply_rtd_convolution(df['RW_NTU'], rtd_weights)
        print(f"[RTD] RTD-convolved RW_NTU created.")
else:
    print(f"[WARNING] FILT_NTU not found, skipping RTD convolution.")
    df['FILT_NTU_RTD'] = np.nan

# ============================================================
# 5. Feature Engineering
# ============================================================
print("\n[Feature Engineering] Creating features...")

TARGET = 'CW_NTU'

# 5a. Time features
df['HOUR'] = pd.to_datetime(df['TIME'], format='%H:%M', errors='coerce').dt.hour
df['MONTH'] = pd.to_datetime(df['DATE'], errors='coerce').dt.month
df['WEEKDAY'] = pd.to_datetime(df['DATE'], errors='coerce').dt.weekday

# Cyclical encoding
df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)

# 5b. Lag features
LAG_VARS = ['RW_NTU', 'RW_PH', 'RW_FLOW', 'RW_CLR', 'FILT_NTU', 'ALUM', 'CL2', 'RIVERLEVEL']
LAG_STEPS = [1, 2, 3, 6, 12]

for var in LAG_VARS:
    if var in df.columns:
        for lag in LAG_STEPS:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

# 5c. Rolling statistics
ROLL_VARS = ['RW_NTU', 'FILT_NTU', 'RIVERLEVEL']
for var in ROLL_VARS:
    if var in df.columns:
        for w in [6, 12]:
            df[f'{var}_roll_mean{w}'] = df[var].rolling(w).mean()

# 5d. RTD-derived features (differences from raw)
if 'FILT_NTU_RTD' in df.columns:
    df['FILT_NTU_RTD_diff'] = df['FILT_NTU'] - df['FILT_NTU_RTD']
    df['FILT_NTU_RTD_ratio'] = df['FILT_NTU'] / (df['FILT_NTU_RTD'] + 1e-10)
if 'RW_NTU_RTD' in df.columns:
    df['RW_NTU_RTD_diff'] = df['RW_NTU'] - df['RW_NTU_RTD']

# Drop NaN rows
initial_rows = len(df)
df.dropna(inplace=True)
print(f"[Feature Engineering] Dropped {initial_rows - len(df)} rows with NaN, remaining: {len(df)}")

# ============================================================
# 6. Prepare data for GBDT
# ============================================================
print("\n" + "=" * 70)
print("Preparing GBDT Model Data")
print("=" * 70)

EXCLUDE_COLS = ['DATETIME', 'DATE', 'TIME', TARGET, 'REMARKS']
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"[Features] Total: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[TARGET].copy()

# Handle inf/nan
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print(f"[Data] Feature matrix: {X.shape}")

# Temporal split (before 2026-02-01 for training)
df_temp = df.copy()
df_temp['DATE_DT'] = pd.to_datetime(df_temp['DATE'], errors='coerce')

train_mask = df_temp['DATE_DT'] < pd.Timestamp('2026-02-01')
test_mask = df_temp['DATE_DT'] >= pd.Timestamp('2026-02-01')

X_train = X[train_mask.values]
y_train = y[train_mask.values]
X_test = X[test_mask.values]
y_test = y[test_mask.values]

print(f"[Split] Training: {len(X_train)} (before 2026-02-01)")
print(f"[Split] Testing:  {len(X_test)} (2026-02-01 onwards)")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 7. GBDT Model Training
# ============================================================
print("\n" + "=" * 70)
print("GBDT Model Training")
print("=" * 70)

gbdt = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features='sqrt',
    loss='huber',  # Robust to outliers
    random_state=42
)

print(f"[GBDT] Training with {gbdt.n_estimators} trees...")
gbdt.fit(X_train_s, y_train)
print(f"[GBDT] Training completed. Best iteration: {gbdt.n_estimators_}")

# Predict
y_train_pred = gbdt.predict(X_train_s)
y_test_pred = gbdt.predict(X_test_s)

# ============================================================
# 8. Model Evaluation
# ============================================================
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n{"=" * 70}")
print("Model Evaluation")
print("=" * 70)
print(f"\n[Training Set]")
print(f"  RMSE = {train_rmse:.4f}")
print(f"  MAE  = {train_mae:.4f}")
print(f"  R2   = {train_r2:.4f}")

print(f"\n[Testing Set]")
print(f"  RMSE = {test_rmse:.4f}")
print(f"  MAE  = {test_mae:.4f}")
print(f"  R2   = {test_r2:.4f}")

# ============================================================
# 9. Predict specific dates (2026-02-01, 02-10, 02-20)
# ============================================================
print("\n" + "=" * 70)
print("Date-specific Predictions")
print("=" * 70)

TARGET_DATES = ['2026-02-01', '2026-02-10', '2026-02-20']
predictions_list = []

for target_date in TARGET_DATES:
    date_mask = df_temp['DATE_DT'] == pd.Timestamp(target_date)
    date_indices = date_mask[date_mask].index

    if len(date_indices) == 0:
        print(f"[Warning] No data found for {target_date}")
        continue

    # Find indices in test set
    valid_test_indices = [date_indices.get_loc(i) for i in date_indices if i in X_test.index]
    if len(valid_test_indices) == 0:
        print(f"[Warning] {target_date} not in test set")
        continue

    actual_vals = y_test.iloc[valid_test_indices].values
    pred_vals = y_test_pred[valid_test_indices]

    hours = df_temp.loc[date_indices[valid_test_indices], 'HOUR'].values if len(valid_test_indices) > 0 else []

    for h, a, p in zip(hours, actual_vals, pred_vals):
        predictions_list.append({
            'Date': target_date,
            'Hour': int(h),
            'Actual_CW_NTU': round(a, 4),
            'Predicted_CW_NTU': round(p, 4),
            'Error': round(a - p, 4)
        })

    print(f"\n  {target_date} predictions:")
    for item in predictions_list:
        if item['Date'] == target_date:
            print(f"    Hour {item['Hour']:2d}: Actual={item['Actual_CW_NTU']:.4f}, "
                  f"Pred={item['Predicted_CW_NTU']:.4f}, Error={item['Error']:.4f}")

# --- Chart 2: Date Predictions ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dates_with_data = []

for ax_idx, target_date in enumerate(TARGET_DATES):
    date_preds = [p for p in predictions_list if p['Date'] == target_date]
    if len(date_preds) == 0:
        axes[ax_idx].text(0.5, 0.5, f'No data for {target_date}', ha='center', va='center', transform=axes[ax_idx].transAxes)
        axes[ax_idx].set_title(f'CW_NTU on {target_date}')
        continue

    hours = [p['Hour'] for p in date_preds]
    actual = [p['Actual_CW_NTU'] for p in date_preds]
    pred = [p['Predicted_CW_NTU'] for p in date_preds]

    axes[ax_idx].plot(hours, actual, 'o-', label='Actual', color='steelblue', linewidth=2, markersize=8)
    axes[ax_idx].plot(hours, pred, 's--', label='Predicted', color='coral', linewidth=2, markersize=8)
    axes[ax_idx].set_xlabel('Hour of Day')
    axes[ax_idx].set_ylabel('CW_NTU')
    axes[ax_idx].set_title(f'CW_NTU on {target_date}')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].set_xticks(hours)
    dates_with_data.append(target_date)

plt.tight_layout()
date_pred_path = os.path.join(OUTPUT_DIR, 'p3_date_predictions.png')
plt.savefig(date_pred_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Chart] Saved: {date_pred_path}")

# ============================================================
# 10. Sensitivity Analysis (OAT - One At a Time)
# ============================================================
print("\n" + "=" * 70)
print("Sensitivity Analysis (OAT Method)")
print("=" * 70)

# Select key features for sensitivity analysis
KEY_FEATURES = ['RW_NTU_lag1', 'RW_NTU_RTD', 'FILT_NTU_RTD', 'FILT_NTU_lag1', 'ALUM_lag1',
                'RW_FLOW_lag1', 'RIVERLEVEL_lag1', 'RW_PH_lag1']
available_key_features = [f for f in KEY_FEATURES if f in feature_cols]

print(f"[Sensitivity] Analyzing {len(available_key_features)} key features: {available_key_features}")

# Use a baseline sample (mean of training data)
baseline_sample = X_train_s.mean(axis=0).reshape(1, -1)
baseline_pred = gbdt.predict(baseline_sample)[0]
print(f"[Sensitivity] Baseline prediction (mean input): {baseline_pred:.6f}")

# Perturb each feature by +/- 10%, 20%, 50%
perturbations = [-0.50, -0.20, -0.10, 0.10, 0.20, 0.50]
sensitivity_results = []

for feat_name in available_key_features:
    feat_idx = feature_cols.index(feat_name)
    feat_std = X_train_s[:, feat_idx].std()
    feat_mean = X_train_s[:, feat_idx].mean()

    for pert in perturbations:
        perturbed = baseline_sample.copy()
        perturbed[0, feat_idx] += pert  # Perturb in standardized space
        new_pred = gbdt.predict(perturbed)[0]
        change = new_pred - baseline_pred
        pct_change = (change / (baseline_pred + 1e-10)) * 100

        sensitivity_results.append({
            'Feature': feat_name,
            'Perturbation': pert,
            'Prediction': round(new_pred, 6),
            'Change': round(change, 6),
            'Pct_Change': round(pct_change, 4)
        })

sens_df = pd.DataFrame(sensitivity_results)

# Summarize sensitivity by feature (average absolute % change)
sens_summary = sens_df.groupby('Feature')['Pct_Change'].agg(['mean', 'std', 'max']).abs()
sens_summary.sort_values('mean', ascending=False, inplace=True)
print(f"\n[Sensitivity] Feature ranking (by mean absolute % change):")
for feat, row in sens_summary.iterrows():
    print(f"  {feat}: {row['mean']:.4f}% (+/- {row['std']:.4f}%)")

# --- Chart 3: Sensitivity Heatmap ---
fig, ax = plt.subplots(figsize=(12, 6))

pivot = sens_df.pivot_table(index='Feature', columns='Perturbation', values='Change', aggfunc='first')
pivot = pivot.reindex(sens_summary.index)

im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-np.abs(pivot.values).max(), vmax=np.abs(pivot.values).max())
ax.set_xticks(range(len(perturbations)))
ax.set_xticklabels([f'{p*100:+.0f}%' for p in perturbations])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_xlabel('Perturbation')
ax.set_ylabel('Feature')
ax.set_title('OAT Sensitivity Analysis: Change in CW_NTU Prediction')

# Add text annotations
for i in range(len(pivot.index)):
    for j in range(len(perturbations)):
        val = pivot.values[i, j]
        text_color = 'white' if abs(val) > np.abs(pivot.values).max() * 0.5 else 'black'
        ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=7, color=text_color)

plt.colorbar(im, ax=ax, label='Change in CW_NTU')
plt.tight_layout()
sens_path = os.path.join(OUTPUT_DIR, 'p3_sensitivity.png')
plt.savefig(sens_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {sens_path}")

# --- Chart 4: Prediction Time Series (full test set) ---
fig, ax = plt.subplots(figsize=(16, 6))

plot_window = min(1000, len(y_test))
test_indices = np.arange(plot_window)

ax.fill_between(test_indices, y_test.values[:plot_window], alpha=0.3, color='steelblue', label='Actual CW_NTU')
ax.plot(test_indices, y_test_pred[:plot_window], 'r-', linewidth=1.5, label='GBDT Predicted', alpha=0.8)
ax.set_xlabel('Time Step (test set)')
ax.set_ylabel('CW_NTU')
ax.set_title(f'RTD-GBDT Model: CW_NTU Time Series Prediction (Test Set, R2={test_r2:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
ts_path = os.path.join(OUTPUT_DIR, 'p3_prediction_timeseries.png')
plt.savefig(ts_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {ts_path}")

# ============================================================
# 11. Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

# Save predictions
pred_df = pd.DataFrame(predictions_list)
pred_path = os.path.join(OUTPUT_DIR, 'problem3_predictions.xlsx')
pred_df.to_excel(pred_path, index=False, engine='openpyxl')
print(f"[Results] Saved date predictions to: {pred_path}")

# Save sensitivity results
sens_path_csv = os.path.join(OUTPUT_DIR, 'problem3_sensitivity.xlsx')
sens_df.to_excel(sens_path_csv, index=False, engine='openpyxl')
print(f"[Results] Saved sensitivity analysis to: {sens_path_csv}")

# Save model metrics
metrics_df = pd.DataFrame({
    'Metric': ['Train_RMSE', 'Train_MAE', 'Train_R2', 'Test_RMSE', 'Test_MAE', 'Test_R2',
               'RTD_N_CSTR', 'RTD_TAU_HOURS', 'RTD_WINDOW_STEPS', 'GBDT_TREES'],
    'Value': [train_rmse, train_mae, train_r2, test_rmse, test_mae, test_r2,
              N_CSTR, TAU_HOURS, rtd_window, gbdt.n_estimators_]
})
metrics_path = os.path.join(OUTPUT_DIR, 'problem3_metrics.xlsx')
metrics_df.to_excel(metrics_path, index=False, engine='openpyxl')
print(f"[Results] Saved model metrics to: {metrics_path}")

# Save feature importance
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gbdt.feature_importances_
}).sort_values('Importance', ascending=False)
feat_imp_path = os.path.join(OUTPUT_DIR, 'problem3_feature_importance.xlsx')
feat_imp.to_excel(feat_imp_path, index=False, engine='openpyxl')
print(f"[Results] Saved feature importance to: {feat_imp_path}")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY - Problem 3: RTD-GBDT Hybrid Prediction Model")
print("=" * 70)
print(f"  RTD Configuration: N={N_CSTR} CSTR, tau={TAU_HOURS}h")
print(f"  RTD Convolution applied to: FILT_NTU, RW_NTU")
print(f"  GBDT trees: {gbdt.n_estimators_}")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")
print(f"  Training R2:  {train_r2:.4f}")
print(f"  Testing R2:   {test_r2:.4f}")
print(f"  Training RMSE: {train_rmse:.4f}")
print(f"  Testing RMSE:  {test_rmse:.4f}")
print(f"  Top 3 features:")
top3 = feat_imp.head(3)
for _, row in top3.iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.4f}")
print(f"  Prediction dates: {', '.join(TARGET_DATES)}")
print(f"  Output files saved to: {OUTPUT_DIR}")
print("=" * 70)
print("Problem 3 completed successfully.")
