# %% 问题3：RTD-GBDT 物理+数据混合动态预测
"""
2020 CUMCM A题：自来水厂水质预测与评估
问题3 —— 结合停留时间分布(RTD)理论构建混合模型，预测未来12h出厂水NTU
可独立运行：python solve_problem3.py
"""
import sys, os, time, math
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm; import matplotlib.pyplot as plt
fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
prop = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings; warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', 'result')
FIG_DIR = os.path.join(RESULT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
TARGET = 'CW_NTU'
TAU = 4.0       # 平均水力停留时间 (h)
N_TANKS = 3     # CSTR串联级数
WINDOW = 12     # RTD卷积窗口

# ========== RTD权重函数 ==========
def rtd_weights(tau=TAU, n=N_TANKS, H=WINDOW):
    t = np.arange(1, H + 1)
    w = (n / tau)**n * t**(n - 1) * np.exp(-n * t / tau) / math.factorial(n - 1)
    return w / w.sum()

# ========== 主流程 ==========
def main():
    t_start = time.time()
    print('=' * 60)
    print('[3/4] 问题3：RTD-GBDT混合动态预测')
    print('=' * 60)

    df = pd.read_csv(os.path.join(RESULT_DIR, 'cleaned_data.csv'))
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # --- RTD卷积特征 ---
    rtd_w = rtd_weights()
    data = df.copy()
    for col in ['FILT_NTU', 'RW_NTU', 'RW_FLOW', 'ALUM']:
        if col in data.columns:
            data[col] = data[col].fillna(method='ffill')
            conv = np.convolve(data[col].values, rtd_w, mode='same')
            data[f'{col}_RTD'] = conv

    data['HOUR'] = data['DATETIME'].dt.hour
    data['MONTH'] = data['DATETIME'].dt.month
    for lag in [1, 2, 3]:
        data[f'TARGET_LAG{lag}'] = data[TARGET].shift(lag)
        for col in ['FILT_NTU', 'RW_NTU']:
            if col in data.columns: data[f'{col}_LAG{lag}'] = data[col].shift(lag)
    for col in data.columns:
        if data[col].isnull().any():
            data[col] = data[col].fillna(method='ffill').fillna(0)

    feats = ['FILT_NTU_RTD', 'RW_NTU_RTD', 'RW_FLOW_RTD', 'ALUM_RTD',
             'FILT_NTU', 'RW_NTU', 'HOUR', 'MONTH',
             'TARGET_LAG1', 'TARGET_LAG2', 'TARGET_LAG3',
             'FILT_NTU_LAG1', 'FILT_NTU_LAG2', 'RW_NTU_LAG1', 'RW_NTU_LAG2']
    feats = [f for f in feats if f in data.columns]
    model_data = data[feats + [TARGET, 'DATETIME']].dropna()
    X = model_data[feats].values; y = model_data[TARGET].values

    n_train = int(len(X) * 0.8)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    # --- GBDT训练 ---
    gbdt = GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbdt.fit(X_tr_s, y_tr)
    y_pred = gbdt.predict(X_te_s)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f'RTD-GBDT 测试集: R² = {r2:.4f}, RMSE = {rmse:.4f} NTU')

    # --- 预测关键日期 ---
    print('\n2026年2月关键日期 7:00-19:00 预测:')
    pred_rows = []
    for date_str in ['2026-02-01', '2026-02-10', '2026-02-20']:
        dt = pd.to_datetime(date_str)
        dd = model_data[(model_data['DATETIME'] >= dt - pd.Timedelta(days=2)) &
                        (model_data['DATETIME'] <= dt + pd.Timedelta(days=2))]
        if len(dd) == 0: continue
        date_preds = []
        for hour in range(7, 20):
            hd = dd[dd['DATETIME'].dt.hour == hour]
            if len(hd) == 0: hd = dd.iloc[[0]]
            row = hd.iloc[0]; Xp = row[feats].values.reshape(1, -1)
            pv = max(0, gbdt.predict(sc.transform(Xp))[0])
            av = row[TARGET] if TARGET in row.index else None
            pred_rows.append({'日期': date_str, '时间': f'{hour:02d}:00',
                'RTD-GBDT预测_NTU': round(pv, 4),
                '实际值_NTU': round(av, 4) if av is not None and not np.isnan(av) else 'N/A'})
            date_preds.append(pv)
        print(f'  {date_str}: 均值={np.mean(date_preds):.4f} NTU')
    pd.DataFrame(pred_rows).to_excel(
        os.path.join(RESULT_DIR, 'problem3_predictions.xlsx'), index=False)

    # --- OAT敏感性分析 ---
    mid_idx = n_train + len(y_te) // 2
    X_base = X_te_s[len(y_te)//2:len(y_te)//2 + 1].copy()
    sens = {}
    for i, f in enumerate(feats[:8]):
        X_pert = X_base.copy(); X_pert[0, i] = X_base[0, i] * 1.1
        pb = gbdt.predict(X_base)[0]; pp = gbdt.predict(X_pert)[0]
        sens[f] = abs(pp - pb) / (abs(pb) + 1e-6)
    print('\nOAT敏感性排名:')
    for f, s in sorted(sens.items(), key=lambda x: -x[1])[:6]:
        print(f'  {f}: {s:.4f}')

    # --- 绘图 ---
    # 图1：RTD分布曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    tv = np.arange(1, 25); wv = rtd_weights(tau=TAU, n=N_TANKS, H=24)
    ax.fill_between(tv, wv, alpha=0.5, color='#3182CE')
    ax.plot(tv, wv, 'o-', color='#3182CE', lw=2, ms=6)
    ax.axvline(x=TAU, color='red', linestyle='--', label=f'τ = {TAU}h')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('清水池停留时间分布 (RTD)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (h)'); ax.set_ylabel('E(t)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p3_rtd_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：关键日期预测
    fig, ax = plt.subplots(figsize=(14, 6))
    cols = ['#3182CE', '#38A169', '#DD6B20']
    for idx, ds in enumerate(['2026-02-01', '2026-02-10', '2026-02-20']):
        dp = [r for r in pred_rows if r['日期'] == ds]
        if dp:
            hh = list(range(7, 20)); vv = [r['RTD-GBDT预测_NTU'] for r in dp]
            ax.plot(hh, vv, 'o-', color=cols[idx], lw=2, ms=6, label=ds)
    ax.axhline(y=1.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='国标 ≤1 NTU')
    ax.set_xticks(range(7, 20)); ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    ax.set_title('问题3：2026年2月关键日期 7:00-19:00 NTU预测', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p3_date_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：敏感性分析
    fig, ax = plt.subplots(figsize=(10, 6))
    fn = [s[0] for s in sorted(sens.items(), key=lambda x: -x[1])[:6]]
    fv = [s[1] for s in sorted(sens.items(), key=lambda x: -x[1])[:6]]
    ax.barh(fn, fv, color=plt.cm.Blues(np.linspace(0.4, 0.95, len(fn))), edgecolor='white')
    ax.set_xlabel('敏感性指数'); ax.set_title('问题3：OAT敏感性分析', fontsize=14, fontweight='bold')
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p3_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图4：预测时间序列
    fig, ax = plt.subplots(figsize=(14, 6))
    dn = min(300, len(y_te))
    ax.plot(range(dn), y_te[:dn], '-', color='#3182CE', lw=1.5, alpha=0.8, label='实际值')
    ax.plot(range(dn), y_pred[:dn], '-', color='#E53E3E', lw=1.5, alpha=0.8,
            label=f'RTD-GBDT (R²={r2:.4f})')
    ax.fill_between(range(dn), y_pred[:dn]-rmse, y_pred[:dn]+rmse,
                    alpha=0.15, color='#E53E3E')
    ax.axhline(y=1.0, color='red', linestyle='--', lw=1, alpha=0.5, label='国标 ≤1 NTU')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    ax.set_title('问题3：RTD-GBDT预测效果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p3_prediction_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t_start
    print(f'\n问题3完成! 总耗时: {elapsed:.1f}s')

if __name__ == '__main__':
    main()
