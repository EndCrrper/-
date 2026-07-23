# -*- coding: utf-8 -*-
"""
Problem 2: Time-delay NARX Dynamic Model
=========================================
Target: FILT_NTU
Inputs: RW_NTU, RW_PH, ALUM, RW_FLOW
CCF cross-correlation for lag estimation (range 0-48)
Grid search for heterogeneous time lags
NARX neural network (MLPRegressor: 64,32,16 hidden layers)
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

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
print("Problem 2: Time-delay NARX Dynamic Model for FILT_NTU Prediction")
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
# 3. Define target and input variables
# ============================================================
TARGET = 'FILT_NTU'
INPUT_VARS = ['RW_NTU', 'RW_PH', 'ALUM', 'RW_FLOW']

print(f"\n[Variables]")
print(f"  Target (y): {TARGET}")
print(f"  Inputs (X): {INPUT_VARS}")

# Check that all columns exist
for col in [TARGET] + INPUT_VARS:
    if col not in df.columns:
        print(f"[ERROR] Column '{col}' not found in data!")
        raise KeyError(f"Missing column: {col}")

# ============================================================
# 4. CCF (Cross-Correlation Function) Analysis for Lag Detection
# ============================================================
print("\n" + "=" * 70)
print("CCF Cross-Correlation Analysis (lag range: 0-48 steps = 0-96 hours)")
print("=" * 70)

MAX_LAG = 48  # Maximum lag to consider (48 steps = 96 hours at 2h intervals)

# Clean data for CCF
ccf_df = df[[TARGET] + INPUT_VARS].copy()
ccf_df.dropna(inplace=True)
print(f"[CCF] Clean samples for CCF: {len(ccf_df)}")

ccf_results = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, input_var in enumerate(INPUT_VARS):
    print(f"\n[CCF] Analyzing {input_var} -> {TARGET}...")
    y_vals = ccf_df[TARGET].values
    x_vals = ccf_df[input_var].values

    cross_corr = []
    lag_values = list(range(0, MAX_LAG + 1))

    for lag in lag_values:
        if lag == 0:
            corr, _ = pearsonr(x_vals, y_vals)
        else:
            corr, _ = pearsonr(x_vals[:-lag], y_vals[lag:])
        cross_corr.append(corr)

    # Find optimal lag (lag with maximum absolute correlation)
    cross_corr = np.array(cross_corr)
    best_lag_idx = np.argmax(np.abs(cross_corr))
    best_lag = lag_values[best_lag_idx]
    best_corr = cross_corr[best_lag_idx]

    ccf_results[input_var] = {
        'lags': lag_values,
        'correlations': cross_corr.tolist(),
        'optimal_lag': best_lag,
        'optimal_correlation': best_corr
    }

    print(f"  Optimal lag: {best_lag} steps ({best_lag * 2}h), correlation: {best_corr:.4f}")

    # Plot CCF
    axes[idx].bar(lag_values, cross_corr, width=0.8, color='steelblue', alpha=0.7)
    axes[idx].axvline(x=best_lag, color='red', linestyle='--', linewidth=2, label=f'Optimal lag={best_lag}')
    axes[idx].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[idx].set_xlabel('Lag (steps, 1 step = 2 hours)')
    axes[idx].set_ylabel('Cross-correlation')
    axes[idx].set_title(f'{input_var} -> {TARGET} (Best lag={best_lag}, r={best_corr:.3f})')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
ccf_path = os.path.join(OUTPUT_DIR, 'p2_ccf_lag.png')
plt.savefig(ccf_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Chart] Saved: {ccf_path}")

# ============================================================
# 5. Grid Search for Heterogeneous Time Lags
# ============================================================
print("\n" + "=" * 70)
print("Grid Search for Heterogeneous Time Lags")
print("=" * 70)

# Define search ranges for each variable (around the CCF-optimal lag)
LAG_SEARCH_RANGES = {}
for var in INPUT_VARS:
    opt_lag = ccf_results[var]['optimal_lag']
    # Search +/- 8 steps around optimum, clamped to [0, MAX_LAG]
    low = max(0, opt_lag - 8)
    high = min(MAX_LAG, opt_lag + 8)
    LAG_SEARCH_RANGES[var] = list(range(low, high + 1))
    print(f"  {var}: optimal lag={opt_lag}, search range=[{low}, {high}]")

# Reduce the search to a subset of candidate lags for practical grid search
# Use: optimal, optimal-4, optimal-2, optimal+2, optimal+4 (clamped)
def get_candidate_lags(opt_lag):
    candidates = set()
    candidates.add(opt_lag)
    for offset in [1, 2, 4, 6, 8]:
        if opt_lag - offset >= 0:
            candidates.add(opt_lag - offset)
        if opt_lag + offset <= MAX_LAG:
            candidates.add(opt_lag + offset)
    return sorted(list(candidates))

candidate_lags = {}
for var in INPUT_VARS:
    candidate_lags[var] = get_candidate_lags(ccf_results[var]['optimal_lag'])
    print(f"  {var} candidate lags: {candidate_lags[var]}")

# Prepare lagged dataset with simple mean imputation
lag_df = df[[TARGET] + INPUT_VARS].copy()
lag_df.dropna(inplace=True)

best_rmse = float('inf')
best_lag_config = None
best_model = None
best_scaler = None

print(f"\n[Grid Search] Searching over lag combinations...")
total_combinations = 1
for var in INPUT_VARS:
    total_combinations *= len(candidate_lags[var])
print(f"  Total combinations: {total_combinations}")

# If too many combinations, use random search
MAX_COMBINATIONS = 500
if total_combinations > MAX_COMBINATIONS:
    print(f"  Too many combinations, using random search ({MAX_COMBINATIONS} samples)")

    import random
    random.seed(42)
    search_count = 0
    for _ in range(MAX_COMBINATIONS * 2):
        if search_count >= MAX_COMBINATIONS:
            break
        lag_config = {}
        for var in INPUT_VARS:
            lag_config[var] = random.choice(candidate_lags[var])

        # Build lagged dataset
        X_list = []
        for var in INPUT_VARS:
            shifted = lag_df[var].shift(lag_config[var])
            X_list.append(shifted)
        X_lagged = pd.concat(X_list, axis=1)
        X_lagged.columns = [f'{var}_lag{lag_config[var]}' for var in INPUT_VARS]
        y_lagged = lag_df[TARGET]

        # Drop NaN
        combined = X_lagged.copy()
        combined['y'] = y_lagged
        combined.dropna(inplace=True)
        X_clean = combined.drop('y', axis=1)
        y_clean = combined['y']

        if len(X_clean) < 100:
            continue

        # Split
        split_idx = int(len(X_clean) * 0.8)
        X_tr, X_te = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_tr, y_te = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train simple model for evaluation (small MLP)
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            verbose=False
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_lag_config = lag_config.copy()
            search_count += 1

else:
    # Exhaustive grid search
    from itertools import product

    search_count = 0
    for lags in product(*[candidate_lags[var] for var in INPUT_VARS]):
        lag_config = dict(zip(INPUT_VARS, lags))

        # Build lagged dataset
        X_list = []
        for var in INPUT_VARS:
            shifted = lag_df[var].shift(lag_config[var])
            X_list.append(shifted)
        X_lagged = pd.concat(X_list, axis=1)
        X_lagged.columns = [f'{var}_lag{lag_config[var]}' for var in INPUT_VARS]
        y_lagged = lag_df[TARGET]

        # Drop NaN
        combined = X_lagged.copy()
        combined['y'] = y_lagged
        combined.dropna(inplace=True)
        X_clean = combined.drop('y', axis=1)
        y_clean = combined['y']

        if len(X_clean) < 100:
            continue

        # Split
        split_idx = int(len(X_clean) * 0.8)
        X_tr, X_te = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_tr, y_te = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train simple model for evaluation
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            verbose=False
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_lag_config = lag_config.copy()
            search_count += 1

print(f"\n[Grid Search] Evaluated {search_count} configurations")
print(f"[Grid Search] Best lag configuration:")
for var, lag in best_lag_config.items():
    print(f"  {var} -> lag {lag} steps ({lag * 2}h)")
print(f"[Grid Search] Best validation RMSE: {best_rmse:.4f}")

# ============================================================
# 6. Build Final NARX Model with Optimal Lags
# ============================================================
print("\n" + "=" * 70)
print("Building NARX Neural Network Model")
print("=" * 70)

# Add auto-regressive lags of FILT_NTU itself (NARX structure)
# Include FILT_NTU lags 1, 2, 3, 6, 12 (2h, 4h, 6h, 12h, 24h)
AUTOREG_LAGS = [1, 2, 3, 6, 12]
X_list_narx = []

# Add exogenous inputs with optimal lags
for var in INPUT_VARS:
    shifted = lag_df[var].shift(best_lag_config[var])
    X_list_narx.append(shifted)

# Add auto-regressive lags of target
for lag in AUTOREG_LAGS:
    shifted = lag_df[TARGET].shift(lag)
    X_list_narx.append(shifted)

X_narx = pd.concat(X_list_narx, axis=1)
col_names = [f'{var}_lag{best_lag_config[var]}' for var in INPUT_VARS]
col_names += [f'{TARGET}_autoreg{lag}' for lag in AUTOREG_LAGS]
X_narx.columns = col_names
y_narx = lag_df[TARGET]

print(f"[NARX] Feature matrix shape: {X_narx.shape}")
print(f"[NARX] Features: {col_names}")

# Drop NaN
combined_narx = X_narx.copy()
combined_narx['y'] = y_narx
combined_narx.dropna(inplace=True)
X_narx_clean = combined_narx.drop('y', axis=1)
y_narx_clean = combined_narx['y']

print(f"[NARX] Clean samples: {len(X_narx_clean)}")

# Temporal split (80/20)
narx_split = int(len(X_narx_clean) * 0.8)
X_narx_train = X_narx_clean.iloc[:narx_split]
X_narx_test = X_narx_clean.iloc[narx_split:]
y_narx_train = y_narx_clean.iloc[:narx_split]
y_narx_test = y_narx_clean.iloc[narx_split:]

print(f"[NARX] Training: {len(X_narx_train)}, Testing: {len(X_narx_test)}")

# Scale features
narx_scaler = StandardScaler()
X_narx_train_s = narx_scaler.fit_transform(X_narx_train)
X_narx_test_s = narx_scaler.transform(X_narx_test)

# Build NARX neural network with 64,32,16 hidden layers
print(f"[NARX] Training MLP with hidden layers: (64, 32, 16)...")
narx_model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    shuffle=True,
    random_state=42,
    tol=1e-4,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

narx_model.fit(X_narx_train_s, y_narx_train)
print(f"[NARX] Training completed in {narx_model.n_iter_} iterations")
print(f"[NARX] Training loss: {narx_model.loss_:.6f}")

# Predict
y_narx_train_pred = narx_model.predict(X_narx_train_s)
y_narx_test_pred = narx_model.predict(X_narx_test_s)

# ============================================================
# 7. Model Evaluation
# ============================================================
print("\n" + "=" * 70)
print("Model Evaluation")
print("=" * 70)

# Training metrics
train_rmse = np.sqrt(mean_squared_error(y_narx_train, y_narx_train_pred))
train_mae = mean_absolute_error(y_narx_train, y_narx_train_pred)
train_r2 = r2_score(y_narx_train, y_narx_train_pred)
train_mape = np.mean(np.abs((y_narx_train.values - y_narx_train_pred) / (y_narx_train.values + 1e-10))) * 100

print(f"\n[Training Set]")
print(f"  RMSE = {train_rmse:.4f}")
print(f"  MAE  = {train_mae:.4f}")
print(f"  R2   = {train_r2:.4f}")
print(f"  MAPE = {train_mape:.2f}%")

# Testing metrics
test_rmse = np.sqrt(mean_squared_error(y_narx_test, y_narx_test_pred))
test_mae = mean_absolute_error(y_narx_test, y_narx_test_pred)
test_r2 = r2_score(y_narx_test, y_narx_test_pred)
test_mape = np.mean(np.abs((y_narx_test.values - y_narx_test_pred) / (y_narx_test.values + 1e-10))) * 100

print(f"\n[Testing Set]")
print(f"  RMSE = {test_rmse:.4f}")
print(f"  MAE  = {test_mae:.4f}")
print(f"  R2   = {test_r2:.4f}")
print(f"  MAPE = {test_mape:.2f}%")

# ============================================================
# 8. Generate Charts
# ============================================================
print("\n" + "=" * 70)
print("Generating Charts")
print("=" * 70)

# --- Chart 1: CCF Lag Analysis (saved earlier, reloading info for the optimal lags chart) ---
fig, ax = plt.subplots(figsize=(12, 6))

lags_display = list(range(0, MAX_LAG + 1))
colors = ['steelblue', 'coral', 'seagreen', 'orange']
for idx, var in enumerate(INPUT_VARS):
    corr_vals = ccf_results[var]['correlations']
    ax.plot(lags_display, corr_vals, color=colors[idx], linewidth=1.5, label=var)
    opt_lag = ccf_results[var]['optimal_lag']
    ax.axvline(x=opt_lag, color=colors[idx], linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Lag (steps, 1 step = 2 hours)')
ax.set_ylabel('Cross-correlation')
ax.set_title('CCF Cross-Correlation: Input Variables vs FILT_NTU')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
ccf2_path = os.path.join(OUTPUT_DIR, 'p2_optimal_lags.png')
plt.savefig(ccf2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {ccf2_path}")

# --- Chart 2: NARX Prediction (time series plot) ---
fig, ax = plt.subplots(figsize=(16, 6))

# Plot test set results (show a representative window)
plot_window = min(500, len(y_narx_test))
test_indices = np.arange(plot_window)

ax.plot(test_indices, y_narx_test.values[:plot_window], 'b-', label='Actual FILT_NTU', linewidth=1.5, alpha=0.8)
ax.plot(test_indices, y_narx_test_pred[:plot_window], 'r--', label='NARX Predicted', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Time Step (test set)')
ax.set_ylabel('FILT_NTU')
ax.set_title(f'NARX Model: FILT_NTU Prediction (Test Set, R2={test_r2:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
narx_pred_path = os.path.join(OUTPUT_DIR, 'p2_narx_prediction.png')
plt.savefig(narx_pred_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {narx_pred_path}")

# --- Chart 3: Fit Accuracy (scatter plot) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training scatter
axes[0].scatter(y_narx_train, y_narx_train_pred, alpha=0.5, s=10, c='steelblue')
min_val = min(y_narx_train.min(), y_narx_train_pred.min())
max_val = max(y_narx_train.max(), y_narx_train_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0].set_xlabel('Actual FILT_NTU')
axes[0].set_ylabel('Predicted FILT_NTU')
axes[0].set_title(f'Training Set (R2={train_r2:.4f})')
axes[0].grid(True, alpha=0.3)

# Testing scatter
axes[1].scatter(y_narx_test, y_narx_test_pred, alpha=0.5, s=10, c='coral')
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[1].set_xlabel('Actual FILT_NTU')
axes[1].set_ylabel('Predicted FILT_NTU')
axes[1].set_title(f'Testing Set (R2={test_r2:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fit_path = os.path.join(OUTPUT_DIR, 'p2_fit_accuracy.png')
plt.savefig(fit_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {fit_path}")

# ============================================================
# 9. Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

# Save lag parameters
lag_params = {
    'Variable': [],
    'Optimal_Lag_Steps': [],
    'Optimal_Lag_Hours': [],
    'Optimal_Correlation': [],
    'CCF_Optimal_Lag': [],
    'Final_Lag_Steps': [],
    'Final_Lag_Hours': []
}

for var in INPUT_VARS:
    lag_params['Variable'].append(var)
    lag_params['Optimal_Lag_Steps'].append(ccf_results[var]['optimal_lag'])
    lag_params['Optimal_Lag_Hours'].append(ccf_results[var]['optimal_lag'] * 2)
    lag_params['Optimal_Correlation'].append(round(ccf_results[var]['optimal_correlation'], 4))
    lag_params['CCF_Optimal_Lag'].append(ccf_results[var]['optimal_lag'])
    lag_params['Final_Lag_Steps'].append(best_lag_config.get(var, ccf_results[var]['optimal_lag']))
    lag_params['Final_Lag_Hours'].append(best_lag_config.get(var, ccf_results[var]['optimal_lag']) * 2)

# Add auto-regressive lags info
for lag in AUTOREG_LAGS:
    lag_params['Variable'].append(f'{TARGET}_autoreg')
    lag_params['Optimal_Lag_Steps'].append(lag)
    lag_params['Optimal_Lag_Hours'].append(lag * 2)
    lag_params['Optimal_Correlation'].append('-')
    lag_params['CCF_Optimal_Lag'].append('-')
    lag_params['Final_Lag_Steps'].append(lag)
    lag_params['Final_Lag_Hours'].append(lag * 2)

lag_df_out = pd.DataFrame(lag_params)
lag_path = os.path.join(OUTPUT_DIR, 'problem2_lag_parameters.csv')
lag_df_out.to_csv(lag_path, index=False, encoding='utf-8-sig')
print(f"[Results] Saved lag parameters to: {lag_path}")

# Also save predictions
pred_df = pd.DataFrame({
    'Actual_FILT_NTU_Train': np.concatenate([y_narx_train.values, np.full(len(y_narx_test), np.nan)]),
    'Predicted_FILT_NTU_Train': np.concatenate([y_narx_train_pred, np.full(len(y_narx_test), np.nan)]),
    'Actual_FILT_NTU_Test': np.concatenate([np.full(len(y_narx_train), np.nan), y_narx_test.values]),
    'Predicted_FILT_NTU_Test': np.concatenate([np.full(len(y_narx_train), np.nan), y_narx_test_pred]),
})
pred_excel_path = os.path.join(OUTPUT_DIR, 'problem2_predictions.xlsx')
pred_df.to_excel(pred_excel_path, index=False, engine='openpyxl')
print(f"[Results] Saved predictions to: {pred_excel_path}")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Train_RMSE', 'Train_MAE', 'Train_R2', 'Train_MAPE',
               'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE'],
    'Value': [train_rmse, train_mae, train_r2, train_mape,
              test_rmse, test_mae, test_r2, test_mape]
})
metrics_path = os.path.join(OUTPUT_DIR, 'problem2_metrics.xlsx')
metrics_df.to_excel(metrics_path, index=False, engine='openpyxl')
print(f"[Results] Saved model metrics to: {metrics_path}")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY - Problem 2: Time-delay NARX Dynamic Model")
print("=" * 70)
print(f"  Target: FILT_NTU (Filtered Water Turbidity)")
print(f"  Exogenous Inputs: {', '.join(INPUT_VARS)}")
print(f"  Auto-regressive lags: {AUTOREG_LAGS}")
print(f"  Optimal heterogeneous lags (from grid search):")
for var, lag in best_lag_config.items():
    print(f"    {var}: {lag} steps ({lag*2}h)")
print(f"  NARX architecture: (64, 32, 16) hidden layers")
print(f"  Training samples: {len(X_narx_train)}")
print(f"  Testing samples: {len(X_narx_test)}")
print(f"  Training R2:  {train_r2:.4f}")
print(f"  Testing R2:   {test_r2:.4f}")
print(f"  Training RMSE: {train_rmse:.4f}")
print(f"  Testing RMSE:  {test_rmse:.4f}")
print(f"  Output files saved to: {OUTPUT_DIR}")
print("=" * 70)
print("Problem 2 completed successfully.")
