# -*- coding: utf-8 -*-
"""
Problem 1: Feature selection + CW_NTU prediction (MIFS-XGBoost-Stacking)
=========================================================================
Three-layer feature selection: Mutual Information -> XGBoost importance -> Permutation Importance
Stacking ensemble: RF, GBDT, XGBoost, SVR (base) + Ridge (meta)
Predicts CW_NTU for 2026-02-01, 2026-02-10, 2026-02-20
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
import sys
from datetime import datetime, timedelta
import math
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

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
print("Problem 1: Feature Selection + Stacking Ensemble for CW_NTU Prediction")
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
# 3. Feature Engineering
# ============================================================
print("\n[Feature Engineering] Creating features...")

# 3a. Lag features for key process variables
LAG_VARS = ['RW_NTU', 'RW_PH', 'RW_CLR', 'RW_FLOW', 'FILT_NTU', 'ALUM', 'CL2', 'RIVERLEVEL']
LAG_STEPS = [1, 2, 3, 6, 12, 24]  # 2h, 4h, 6h, 12h, 24h, 48h

for var in LAG_VARS:
    if var in df.columns:
        for lag in LAG_STEPS:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

# 3b. Rolling window statistics (mean, std) for key process variables
ROLL_VARS = ['RW_NTU', 'RW_PH', 'FILT_NTU', 'RIVERLEVEL']
ROLL_WINDOWS = [6, 12, 24]  # 12h, 24h, 48h

for var in ROLL_VARS:
    if var in df.columns:
        for w in ROLL_WINDOWS:
            df[f'{var}_roll_mean{w}'] = df[var].rolling(w).mean()
            df[f'{var}_roll_std{w}'] = df[var].rolling(w).std()

# 3c. Time features
df['HOUR'] = pd.to_datetime(df['TIME'], format='%H:%M', errors='coerce').dt.hour
df['MONTH'] = pd.to_datetime(df['DATE'], errors='coerce').dt.month
df['DAYOFYEAR'] = pd.to_datetime(df['DATE'], errors='coerce').dt.dayofyear
df['WEEKDAY'] = pd.to_datetime(df['DATE'], errors='coerce').dt.weekday

# Cyclical encoding for hour (daily seasonality)
df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)

# Cyclical encoding for month (annual seasonality)
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)

# 3d. Rate of change features (derivative)
if 'RW_NTU' in df.columns:
    df['RW_NTU_diff'] = df['RW_NTU'].diff()
    df['RW_NTU_diff_abs'] = df['RW_NTU_diff'].abs()
if 'FILT_NTU' in df.columns:
    df['FILT_NTU_diff'] = df['FILT_NTU'].diff()

# 3e. Interaction features
if 'ALUM' in df.columns and 'RW_NTU' in df.columns:
    df['ALUM_RW_NTU'] = df['ALUM'] * df['RW_NTU']
if 'ALUM' in df.columns and 'RW_FLOW' in df.columns:
    df['ALUM_FLOW'] = df['ALUM'] * df['RW_FLOW']

# Drop NaN rows from feature creation
initial_rows = len(df)
df.dropna(inplace=True)
print(f"[Feature Engineering] Dropped {initial_rows - len(df)} rows with NaN, remaining: {len(df)}")

# ============================================================
# 4. Define features and target
# ============================================================
TARGET = 'CW_NTU'
print(f"\n[Model] Target: {TARGET}")

# Exclude non-feature columns
EXCLUDE_COLS = ['DATETIME', 'DATE', 'TIME', TARGET, 'REMARKS']
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"[Model] Total candidate features: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[TARGET].copy()

# Handle any remaining inf values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print(f"[Model] Feature matrix shape: {X.shape}")
print(f"[Model] Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

# ============================================================
# 5. Split data (temporal split)
# ============================================================
# Use data before 2026-02-01 for training/validation
df_temp = df.copy()
df_temp['DATE_DT'] = pd.to_datetime(df_temp['DATE'], errors='coerce')

train_mask = df_temp['DATE_DT'] < pd.Timestamp('2026-02-01')
test_mask = df_temp['DATE_DT'] >= pd.Timestamp('2026-02-01')

X_train = X[train_mask.values]
y_train = y[train_mask.values]
X_test = X[test_mask.values]
y_test = y[test_mask.values]

print(f"\n[Split] Training: {len(X_train)} samples (before 2026-02-01)")
print(f"[Split] Testing: {len(X_test)} samples (2026-02-01 onwards)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = list(feature_cols)

# ============================================================
# 6. Three-layer Feature Selection
# ============================================================
print("\n" + "=" * 70)
print("Layer 1: Mutual Information Feature Selection")
print("=" * 70)

mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

# Keep top 50% by MI score
mi_threshold = np.percentile(mi_scores, 50)
mi_selected = feature_names[np.array(mi_scores) >= mi_threshold]
print(f"[MI] Threshold (50th percentile): {mi_threshold:.4f}")
print(f"[MI] Selected {len(mi_selected)} features out of {len(feature_names)}")

print("\n" + "=" * 70)
print("Layer 2: XGBoost Importance Selection")
print("=" * 70)

# Train XGBoost on MI-selected features
mi_indices = [feature_names.index(f) for f in mi_selected]
X_train_mi = X_train_scaled[:, mi_indices]
X_test_mi = X_test_scaled[:, mi_indices]

xgb_selector = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_selector.fit(X_train_mi, y_train)

xgb_imp = pd.Series(xgb_selector.feature_importances_, index=mi_selected).sort_values(ascending=False)

# Keep features with cumulative importance up to 95%
xgb_imp_sorted = xgb_imp.sort_values(ascending=False)
cumsum = xgb_imp_sorted.cumsum()
xgb_selected = xgb_imp_sorted[cumsum <= 0.95].index.tolist()
# Ensure at least one feature
if len(xgb_selected) == 0:
    xgb_selected = [xgb_imp_sorted.index[0]]
print(f"[XGBoost] Selected {len(xgb_selected)} features (cumulative importance <= 95%)")

print("\n" + "=" * 70)
print("Layer 3: Permutation Importance Selection")
print("=" * 70)

# Train RF on XGBoost-selected features
xgb_indices = [list(mi_selected).index(f) for f in xgb_selected]
X_train_xgb = X_train_mi[:, xgb_indices]
X_test_xgb = X_test_mi[:, xgb_indices]

rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_xgb, y_train)

perm_result = permutation_importance(
    rf_selector, X_train_xgb, y_train,
    n_repeats=5, random_state=42, n_jobs=-1
)
perm_imp = pd.Series(perm_result.importances_mean, index=xgb_selected).sort_values(ascending=False)

# Keep features with positive permutation importance
perm_selected = perm_imp[perm_imp > 0].index.tolist()
print(f"[Permutation] Selected {len(perm_selected)} features (positive importance)")

# Final feature set
final_features = list(perm_selected)
final_indices = [list(mi_selected).index(f) for f in final_features]
print(f"\n[Final] Selected {len(final_features)} features for modeling")
print(f"[Final] Features: {final_features}")

# ============================================================
# 7. Prepare final training data
# ============================================================
X_train_final = X_train_mi[:, final_indices]
X_test_final = X_test_mi[:, final_indices]

print(f"\n[Data] Final training set: {X_train_final.shape}")
print(f"[Data] Final testing set: {X_test_final.shape}")

# ============================================================
# 8. Stacking Ensemble Model
# ============================================================
print("\n" + "=" * 70)
print("Stacking Ensemble: RF + GBDT + XGBoost + SVR -> Ridge Meta")
print("=" * 70)

# Base models
base_models = {
    'RF': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'GBDT': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, verbosity=0),
    'SVR': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.01)
}

# Meta model
meta_model = Ridge(alpha=1.0)

# Train base models and collect out-of-fold predictions
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Store predictions for meta training
meta_train_features = np.zeros((X_train_final.shape[0], len(base_models)))
meta_test_features = np.zeros((X_test_final.shape[0], len(base_models)))

# Store individual model performance
model_scores = {}

for i, (name, model) in enumerate(base_models.items()):
    print(f"\n[Base Model] Training {name}...")
    oof_pred = np.zeros(X_train_final.shape[0])
    test_pred = np.zeros(X_test_final.shape[0])

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final)):
        X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        oof_pred[val_idx] = model_clone.predict(X_val)

        # Test predictions (average across folds)
        test_pred += model_clone.predict(X_test_final) / N_FOLDS

    meta_train_features[:, i] = oof_pred
    meta_test_features[:, i] = test_pred

    # Evaluate OOF performance
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_pred))
    oof_mae = mean_absolute_error(y_train, oof_pred)
    oof_r2 = r2_score(y_train, oof_pred)
    model_scores[name] = {'RMSE': oof_rmse, 'MAE': oof_mae, 'R2': oof_r2}
    print(f"  [{name}] OOF - RMSE={oof_rmse:.4f}, MAE={oof_mae:.4f}, R2={oof_r2:.4f}")

# Train meta model
print(f"\n[Meta Model] Training Ridge regression...")
meta_model.fit(meta_train_features, y_train)
meta_train_pred = meta_model.predict(meta_train_features)
meta_test_pred = meta_model.predict(meta_test_features)

print(f"[Meta Model] Meta coefficients: {dict(zip(base_models.keys(), meta_model.coef_))}")

# Ensemble predictions
y_train_pred = meta_train_pred
y_test_pred = meta_test_pred

# ============================================================
# 9. Model Evaluation
# ============================================================
print("\n" + "=" * 70)
print("Model Evaluation")
print("=" * 70)

# Training metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_ev = explained_variance_score(y_train, y_train_pred)

print(f"\n[Training Set]")
print(f"  RMSE = {train_rmse:.4f}")
print(f"  MAE  = {train_mae:.4f}")
print(f"  R2   = {train_r2:.4f}")
print(f"  EV   = {train_ev:.4f}")

# Testing metrics
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_ev = explained_variance_score(y_test, y_test_pred)

print(f"\n[Testing Set]")
print(f"  RMSE = {test_rmse:.4f}")
print(f"  MAE  = {test_mae:.4f}")
print(f"  R2   = {test_r2:.4f}")
print(f"  EV   = {test_ev:.4f}")

# ============================================================
# 10. Generate Charts
# ============================================================
print("\n" + "=" * 70)
print("Generating Charts")
print("=" * 70)

# --- Chart 1: Feature Selection (MI + XGBoost + Permutation) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MI scores (top 15)
top_mi = mi_series.head(15)
axes[0].barh(range(len(top_mi)), top_mi.values, color='steelblue')
axes[0].set_yticks(range(len(top_mi)))
axes[0].set_yticklabels(top_mi.index, fontsize=8)
axes[0].set_xlabel('Mutual Information Score')
axes[0].set_title('Mutual Information (Top 15)', fontsize=12)
axes[0].invert_yaxis()

# XGBoost importance (top 15)
top_xgb = xgb_imp.head(15)
axes[1].barh(range(len(top_xgb)), top_xgb.values, color='coral')
axes[1].set_yticks(range(len(top_xgb)))
axes[1].set_yticklabels(top_xgb.index, fontsize=8)
axes[1].set_xlabel('XGBoost Importance')
axes[1].set_title('XGBoost Feature Importance (Top 15)', fontsize=12)
axes[1].invert_yaxis()

# Permutation importance (top 15)
top_perm = perm_imp.head(15)
axes[2].barh(range(len(top_perm)), top_perm.values, color='seagreen')
axes[2].set_yticks(range(len(top_perm)))
axes[2].set_yticklabels(top_perm.index, fontsize=8)
axes[2].set_xlabel('Permutation Importance')
axes[2].set_title('Permutation Importance (Top 15)', fontsize=12)
axes[2].invert_yaxis()

plt.tight_layout()
p1_path = os.path.join(OUTPUT_DIR, 'p1_feature_selection.png')
plt.savefig(p1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {p1_path}")

# --- Chart 2: Model Comparison (base models vs stacking) ---
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(model_scores.keys()) + ['Stacking']
r2_vals = [model_scores[m]['R2'] for m in model_scores.keys()] + [train_r2]
rmse_vals = [model_scores[m]['RMSE'] for m in model_scores.keys()] + [train_rmse]

x_pos = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, r2_vals, width, label='R2 Score', color='steelblue')
bars2 = ax.bar(x_pos + width/2, rmse_vals, width, label='RMSE', color='coral')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (OOF/Stacking)')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.legend()

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
p2_path = os.path.join(OUTPUT_DIR, 'p1_model_comparison.png')
plt.savefig(p2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {p2_path}")

# --- Chart 3: Predicted vs Actual ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=10, c='steelblue')
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0].set_xlabel('Actual CW_NTU')
axes[0].set_ylabel('Predicted CW_NTU')
axes[0].set_title(f'Training Set (R2={train_r2:.4f}, RMSE={train_rmse:.4f})')
axes[0].grid(True, alpha=0.3)

# Testing set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=10, c='coral')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[1].set_xlabel('Actual CW_NTU')
axes[1].set_ylabel('Predicted CW_NTU')
axes[1].set_title(f'Testing Set (R2={test_r2:.4f}, RMSE={test_rmse:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
p3_path = os.path.join(OUTPUT_DIR, 'p1_pred_vs_actual.png')
plt.savefig(p3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {p3_path}")

# --- Chart 4: Residual Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Residuals vs predicted
axes[0, 0].scatter(y_train_pred, residuals_train, alpha=0.5, s=10, c='steelblue')
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=1)
axes[0, 0].set_xlabel('Predicted CW_NTU')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted (Training)')
axes[0, 0].grid(True, alpha=0.3)

# Histogram of residuals
axes[0, 1].hist(residuals_train, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Residual Distribution (Training)')

# Q-Q plot
stats.probplot(residuals_train, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Training Residuals)')
axes[1, 0].grid(True, alpha=0.3)

# Residuals over time (test set)
test_dates = df_temp.loc[test_mask.values, 'DATETIME'].values
axes[1, 1].plot(range(len(residuals_test)), residuals_test, 'o-', markersize=3, linewidth=0.5, color='coral')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=1)
axes[1, 1].set_xlabel('Sample Index (Test Set)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Test Set Residuals Over Time')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
p4_path = os.path.join(OUTPUT_DIR, 'p1_residual_analysis.png')
plt.savefig(p4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {p4_path}")

# --- Chart 5: Date-specific predictions (2026-02-01, 02-10, 02-20) ---
target_dates = ['2026-02-01', '2026-02-10', '2026-02-20']
predictions_dict = {'Date': [], 'Hour': [], 'Actual_CW_NTU': [], 'Predicted_CW_NTU': [], 'Error': []}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, target_date in enumerate(target_dates):
    date_mask = df_temp['DATE_DT'] == pd.Timestamp(target_date)
    date_indices = date_mask[date_mask].index

    if len(date_indices) == 0:
        print(f"[Warning] No data found for {target_date}")
        axes[idx].text(0.5, 0.5, f'No data for {target_date}', ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'Predictions for {target_date}')
        continue

    # Get test indices for this date
    date_test_idx = [i for i, idx_val in enumerate(date_indices) if idx_val in X_test.index]
    if len(date_test_idx) == 0:
        print(f"[Warning] {target_date} not in test set")
        axes[idx].text(0.5, 0.5, f'{target_date} not in test set', ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'Predictions for {target_date}')
        continue

    actual_vals = y_test.iloc[date_test_idx].values
    pred_vals = y_test_pred[date_test_idx]

    # Get hour labels
    hours = []
    for idx_val in date_indices:
        if idx_val in X_test.index:
            hours.append(df_temp.loc[idx_val, 'HOUR'])

    axes[idx].plot(hours, actual_vals, 'o-', label='Actual', color='steelblue', linewidth=2, markersize=6)
    axes[idx].plot(hours, pred_vals, 's--', label='Predicted', color='coral', linewidth=2, markersize=6)
    axes[idx].set_xlabel('Hour of Day')
    axes[idx].set_ylabel('CW_NTU')
    axes[idx].set_title(f'CW_NTU on {target_date}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(hours)

    for h, a, p in zip(hours, actual_vals, pred_vals):
        predictions_dict['Date'].append(target_date)
        predictions_dict['Hour'].append(int(h))
        predictions_dict['Actual_CW_NTU'].append(round(a, 4))
        predictions_dict['Predicted_CW_NTU'].append(round(p, 4))
        predictions_dict['Error'].append(round(a - p, 4))

plt.tight_layout()
p5_path = os.path.join(OUTPUT_DIR, 'p1_date_predictions.png')
plt.savefig(p5_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {p5_path}")

# ============================================================
# 11. Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

# Save date-specific predictions
pred_df = pd.DataFrame(predictions_dict)
pred_path = os.path.join(OUTPUT_DIR, 'problem1_predictions.xlsx')
pred_df.to_excel(pred_path, index=False, engine='openpyxl')
print(f"[Results] Saved date predictions to: {pred_path}")

# Save feature importance summary
feature_importance_df = pd.DataFrame({
    'Feature': mi_series.index,
    'MI_Score': mi_series.values,
    'XGBoost_Importance': xgb_imp.reindex(mi_series.index, fill_value=0).values,
    'Permutation_Importance': perm_imp.reindex(mi_series.index, fill_value=0).values,
    'Selected': mi_series.index.isin(final_features)
})
feat_path = os.path.join(OUTPUT_DIR, 'problem1_feature_importance.xlsx')
feature_importance_df.to_excel(feat_path, index=False, engine='openpyxl')
print(f"[Results] Saved feature importance to: {feat_path}")

# Save model metrics summary
metrics_dict = {
    'Metric': ['Train_RMSE', 'Train_MAE', 'Train_R2', 'Train_EV', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_EV'],
    'Value': [train_rmse, train_mae, train_r2, train_ev, test_rmse, test_mae, test_r2, test_ev]
}
for name, scores in model_scores.items():
    metrics_dict['Metric'].extend([f'{name}_RMSE', f'{name}_MAE', f'{name}_R2'])
    metrics_dict['Value'].extend([scores['RMSE'], scores['MAE'], scores['R2']])

metrics_df = pd.DataFrame(metrics_dict)
metrics_path = os.path.join(OUTPUT_DIR, 'problem1_metrics.xlsx')
metrics_df.to_excel(metrics_path, index=False, engine='openpyxl')
print(f"[Results] Saved model metrics to: {metrics_path}")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY - Problem 1: Feature Selection + Stacking Ensemble")
print("=" * 70)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")
print(f"  Initial features: {len(feature_cols)}")
print(f"  Final selected features: {len(final_features)}")
print(f"  Training RMSE: {train_rmse:.4f}")
print(f"  Training R2:   {train_r2:.4f}")
print(f"  Testing RMSE:  {test_rmse:.4f}")
print(f"  Testing R2:    {test_r2:.4f}")
print(f"  Prediction dates: {', '.join(target_dates)}")
print(f"  Output files saved to: {OUTPUT_DIR}")
print("=" * 70)
print("Problem 1 completed successfully.")
