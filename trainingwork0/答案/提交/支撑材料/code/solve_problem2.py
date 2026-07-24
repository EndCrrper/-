# %% 问题2：TD-NARX 异质时滞动态建模
"""
2020 CUMCM A题：自来水厂水质预测与评估
问题2 —— 估计原水指标和操作变量对滤后水浊度的异质时滞参数
可独立运行：python solve_problem2.py
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm; import matplotlib.pyplot as plt
fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
prop = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product
import warnings; warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', 'result')
FIG_DIR = os.path.join(RESULT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
TARGET = 'FILT_NTU'
INPUT_VARS = ['RW_NTU', 'RW_PH', 'ALUM', 'RW_FLOW']

# ========== 主流程 ==========
def main():
    t_start = time.time()
    print('=' * 60)
    print('[2/4] 问题2：滤后水浊度(FILT.NTU)动态时滞建模')
    print('=' * 60)

    df = pd.read_csv(os.path.join(RESULT_DIR, 'cleaned_data.csv'))
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    data = df[['DATETIME'] + INPUT_VARS + [TARGET]].copy()
    for col in INPUT_VARS + [TARGET]:
        data[col] = data[col].fillna(method='ffill').fillna(data[col].median())

    # --- CCF 互相关时滞估计 ---
    max_lag_ccf = 48
    lag_estimates = {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for idx, var in enumerate(INPUT_VARS):
        x = data[var].values; y = data[TARGET].values
        valid = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[valid], y[valid]
        if xc.std() < 1e-6 or yc.std() < 1e-6:
            lag_estimates[var] = 4
            axes[idx].text(0.5, 0.5, f'{var}\n方差趋近于0\n(默认时滞=4步)',
                ha='center', va='center', transform=axes[idx].transAxes, fontsize=11, color='gray')
            axes[idx].set_title(f'{var} (无方差)')
        else:
            ccf = signal.correlate(yc - yc.mean(), xc - xc.mean(), mode='full')
            ccf = ccf / (len(xc) * xc.std() * yc.std() + 1e-10)
            mid = len(ccf) // 2
            neg_filter = np.arange(-mid, len(ccf) - mid)
            usable_mask = (neg_filter < 0) & (neg_filter >= -max_lag_ccf)
            usable_lags = -neg_filter[usable_mask]
            usable_ccf = np.abs(ccf[usable_mask])
            best_lag = min(usable_lags[np.argmax(usable_ccf)], 24) if len(usable_ccf) > 0 else 4
            lag_estimates[var] = best_lag
            plot_range = slice(mid - max_lag_ccf, mid + max_lag_ccf)
            plot_lags = np.arange(-max_lag_ccf, max_lag_ccf)
            axes[idx].plot(plot_lags, ccf[plot_range], color='#3182CE', linewidth=1)
            axes[idx].axvline(x=-best_lag, color='red', linestyle='--',
                label=f'最优: {best_lag}步 ({best_lag*2}h)')
            axes[idx].axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            axes[idx].legend(fontsize=9)
        axes[idx].set_xlabel('滞后步数'); axes[idx].set_ylabel('CCF')
        axes[idx].grid(True, alpha=0.3)
        print(f'  {var}: CCF最优时滞 = {lag_estimates[var]}步 ({lag_estimates[var]*2}h)')
    fig.suptitle('问题2：CCF互相关时滞估计', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p2_ccf_lag.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grid Search 精搜最优异质时滞 ---
    search_ranges = {}
    for var in INPUT_VARS:
        base = max(1, min(lag_estimates.get(var, 4), 12))
        rng = list(range(max(0, base - 3), min(max_lag_ccf, base + 4)))
        if not rng: rng = [base]
        search_ranges[var] = rng

    np.random.seed(42)
    all_combos = list(product(*search_ranges.values()))
    sample_combos = all_combos if len(all_combos) <= 200 else \
        [all_combos[i] for i in np.random.choice(len(all_combos), 200, replace=False)]

    best_score = -np.inf; best_lags = None
    for lc in sample_combos:
        X_lag, y_lag = [], []
        mn = max(lc)
        for t in range(mn, len(data)):
            row = [data[var].iloc[t - lc[vi]] for vi, var in enumerate(INPUT_VARS)]
            X_lag.append(row); y_lag.append(data[TARGET].iloc[t])
        X_lag = np.array(X_lag); y_lag = np.array(y_lag)
        if len(X_lag) < 100: continue
        lr = LinearRegression()
        lr.fit(X_lag[:int(len(X_lag)*0.8)], y_lag[:int(len(y_lag)*0.8)])
        score = lr.score(X_lag[int(len(X_lag)*0.8):], y_lag[int(len(y_lag)*0.8):])
        if score > best_score: best_score = score; best_lags = lc

    final_lags = {var: (best_lags[vi] if best_lags else lag_estimates.get(var, 4))
                  for vi, var in enumerate(INPUT_VARS)}

    print('\n最优异质时滞 (CCF + GridSearch):')
    for var in INPUT_VARS:
        print(f'  {var}: {final_lags[var]}步 ({final_lags[var]*2}h)')

    # --- NARX 神经网络 ---
    max_lag = max(final_lags.values())
    X_narx, y_narx = [], []
    for t in range(max_lag, len(data)):
        row = [data[var].iloc[t - final_lags[var]] for var in INPUT_VARS]
        for al in [1, 2, 3]:
            if t - al >= 0: row.append(data[TARGET].iloc[t - al])
        X_narx.append(row); y_narx.append(data[TARGET].iloc[t])
    X_narx = np.array(X_narx); y_narx = np.array(y_narx)

    n_train = int(len(X_narx) * 0.8)
    X_tr, X_te = X_narx[:n_train], X_narx[n_train:]
    y_tr, y_te = y_narx[:n_train], y_narx[n_train:]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    narx = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam',
        alpha=0.0001, batch_size=64, learning_rate='adaptive', max_iter=500,
        early_stopping=True, validation_fraction=0.1, random_state=42)
    narx.fit(X_tr_s, y_tr)
    y_pred = narx.predict(X_te_s)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f'\nTD-NARX 测试集: R² = {r2:.4f}, RMSE = {rmse:.4f} NTU')

    # --- 保存时滞参数 ---
    pd.DataFrame([{'变量': v, '时滞(步)': final_lags[v], '时滞(h)': final_lags[v]*2}
                  for v in INPUT_VARS]).to_csv(
        os.path.join(RESULT_DIR, 'problem2_lag_parameters.csv'), index=False, encoding='utf-8-sig')

    # --- 绘图 ---
    # 图1：最优时滞条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    vp = list(final_lags.keys()); lp = [final_lags[v]*2 for v in vp]
    clrs = ['#3182CE', '#38A169', '#DD6B20', '#805AD5']
    bars = ax.barh(vp, lp, color=clrs, height=0.5)
    for b, h in zip(bars, lp):
        ax.text(b.get_width() + 0.2, b.get_y() + b.get_height()/2,
                f'{h}h', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('最优时滞 (小时)'); ax.set_title('问题2：异质时滞估计', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p2_optimal_lags.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：NARX预测效果
    fig, ax = plt.subplots(figsize=(14, 6))
    sn = min(200, len(y_te))
    ax.plot(range(sn), y_te[:sn], 'o-', color='#3182CE', ms=4, lw=1.5, alpha=0.8, label='实际值')
    ax.plot(range(sn), y_pred[:sn], 's-', color='#E53E3E', ms=4, lw=1.5, alpha=0.8,
            label=f'TD-NARX (R²={r2:.3f})')
    ax.fill_between(range(sn), y_pred[:sn]-rmse, y_pred[:sn]+rmse,
                    alpha=0.15, color='#E53E3E', label=f'±RMSE ({rmse:.3f})')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('问题2：TD-NARX预测效果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p2_narx_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：拟合精度
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_te, y_pred, alpha=0.5, s=20, c='#3182CE', edgecolors='white', linewidth=0.5)
    mi_v, ma_v = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
    axes[0].plot([mi_v, ma_v], [mi_v, ma_v], 'r--', linewidth=2)
    axes[0].set_xlabel('实际值'); axes[0].set_ylabel('预测值')
    axes[0].set_title(f'预测 vs 实际 (R²={r2:.4f})'); axes[0].grid(True, alpha=0.3)
    res = y_te - y_pred
    axes[1].hist(res, bins=50, color='#68D391', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_title(f'残差分布 (μ={res.mean():.4f}, σ={res.std():.4f})')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p2_fit_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t_start
    print(f'\n问题2完成! 总耗时: {elapsed:.1f}s')

if __name__ == '__main__':
    main()
