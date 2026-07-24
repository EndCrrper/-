# %% 问题1：MIFS-XGBoost-Stacking 三层递进特征筛选与多模型融合预测
"""
2020 CUMCM A题：自来水厂水质预测与评估
问题1 —— 筛选影响NTU的主要因素，建立预测模型，预测2026年2月关键日期
可独立运行：python solve_problem1.py
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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import warnings; warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', 'result')
FIG_DIR = os.path.join(RESULT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
TARGET = 'CW_NTU'

# ========== 主流程 ==========
def main():
    t_start = time.time()
    print('=' * 60)
    print('[1/4] 问题1：因素筛选与NTU预测 (MIFS-XGBoost-Stacking)')
    print('=' * 60)

    # --- 加载并特征工程 ---
    df = pd.read_csv(os.path.join(RESULT_DIR, 'cleaned_data.csv'))
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['HOUR'] = df['DATETIME'].dt.hour
    df['MONTH'] = df['DATETIME'].dt.month
    df['DAY'] = df['DATETIME'].dt.day
    df['DOW'] = df['DATETIME'].dt.dayofweek

    for col in ['RW_NTU', 'RW_CLR', 'RW_FLOW', 'FILT_NTU', 'ALUM']:
        for w in [6, 12, 24]:
            df[f'{col}_ROLL{w}'] = df[col].rolling(window=w, min_periods=1).mean()
    for col in ['RW_NTU', 'FILT_NTU', 'RW_FLOW', 'ALUM']:
        for lag in [1, 2]:
            df[f'{col}_LAG{lag}'] = df[col].shift(lag)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill').fillna(
                df[col].median() if str(df[col].dtype) in ['float64', 'int64'] else 0)

    base_feats = ['RIVERLEVEL', 'RW_FLOW', 'RW_NTU', 'RW_CLR', 'RW_PH',
        'CW_WELL_LEVEL', 'CW_PH', 'CL2', 'ALUM', 'TW_FLOW', 'FILT_NTU', 'CW_CLR', 'F_RIDE']
    all_feats = [f for f in base_feats + ['HOUR', 'MONTH', 'DAY', 'DOW'] +
        [c for c in df.columns if '_ROLL' in c or '_LAG' in c] if f in df.columns]
    data = df[all_feats + [TARGET, 'DATETIME']].dropna(subset=[TARGET])
    for col in all_feats:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
    X = data[all_feats].values; y = data[TARGET].values

    # --- 三层递进特征筛选 ---
    # 第一层：互信息
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': all_feats, 'MI': mi_scores}).sort_values('MI', ascending=False)
    sel_mi = [f for f, s in zip(all_feats, mi_scores) if s > np.median(mi_scores) * 0.3]

    # 第二层：XGBoost重要性
    xgb = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42).fit(X, y)
    imp_df = pd.DataFrame({'Feature': all_feats, 'Importance': xgb.feature_importances_}).sort_values('Importance', ascending=False)
    cumsum = imp_df['Importance'].cumsum()
    top_xgb = imp_df[cumsum <= 0.92]['Feature'].tolist()
    if len(top_xgb) < 8: top_xgb = imp_df.head(12)['Feature'].tolist()

    # 第三层：排列重要性
    perm = permutation_importance(xgb, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({'Feature': all_feats, 'Importance': perm.importances_mean}).sort_values('Importance', ascending=False)

    # 汇总
    final_feats = list(set(sel_mi) & set(top_xgb) & set(perm_df.head(15)['Feature'].tolist()))
    for f in set(sel_mi) & set(top_xgb):
        if f not in final_feats: final_feats.append(f)
    for f in ['RW_NTU', 'FILT_NTU', 'ALUM', 'RW_PH', 'RW_FLOW']:
        if f in all_feats and f not in final_feats: final_feats.append(f)
    print(f'  三层筛选: MI={len(sel_mi)} -> XGB={len(top_xgb)} -> 最终={len(final_feats)} 个特征')

    # --- Stacking集成 ---
    Xf = data[final_feats].values
    n_train = int(len(Xf) * 0.8)
    X_tr, X_te = Xf[:n_train], Xf[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    def oof_pred(model, X_tr, y_tr, X_te):
        oof = np.zeros(len(X_tr)); te = np.zeros(len(X_te))
        for ti, vi in kf.split(X_tr):
            model.fit(X_tr[ti], y_tr[ti])
            oof[vi] = model.predict(X_tr[vi])
            te += model.predict(X_te) / kf.n_splits
        return oof, te

    rf_oof, rf_te = oof_pred(RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1), X_tr_s, y_tr, X_te_s)
    gbdt_oof, gbdt_te = oof_pred(GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42), X_tr_s, y_tr, X_te_s)
    xgb_oof, xgb_te = oof_pred(XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42), X_tr_s, y_tr, X_te_s)
    svr_oof, svr_te = oof_pred(SVR(kernel='rbf', C=5, gamma='scale', epsilon=0.01), X_tr_s, y_tr, X_te_s)
    meta = Ridge(alpha=0.5).fit(np.column_stack([rf_oof, gbdt_oof, xgb_oof, svr_oof]), y_tr)
    stacking_pred = meta.predict(np.column_stack([rf_te, gbdt_te, xgb_te, svr_te]))

    # 全量训练
    rf_full = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
    gbdt_full = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42).fit(X_tr_s, y_tr)
    xgb_full = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42).fit(X_tr_s, y_tr)
    svr_full = SVR(kernel='rbf', C=5, gamma='scale', epsilon=0.01).fit(X_tr_s, y_tr)

    models = {'RF': rf_full.predict(X_te_s), 'GBDT': gbdt_full.predict(X_te_s),
              'XGBoost': xgb_full.predict(X_te_s), 'SVR': svr_full.predict(X_te_s),
              'Stacking': stacking_pred}

    # --- 输出结果 ---
    print('\n模型性能对比:')
    print(f'  {"模型":12s}  {"R²":>8s}  {"RMSE":>8s}  {"MAE":>8s}')
    print(f'  {"-"*42}')
    results = {}
    for name, pred in models.items():
        r2 = r2_score(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        print(f'  {name:12s}  {r2:8.4f}  {rmse:8.4f}  {mae:8.4f}')

    # --- 预测关键日期 ---
    print('\n2026年2月关键日期预测:')
    pred_rows = []
    for ds in ['2026-02-01', '2026-02-10', '2026-02-20']:
        dt = pd.to_datetime(ds)
        dd = data[(data['DATETIME'] >= dt - pd.Timedelta(days=3)) &
                  (data['DATETIME'] <= dt + pd.Timedelta(days=3))].copy()
        if len(dd) == 0: continue
        Xp = dd[final_feats].values; Xps = sc.transform(Xp)
        rp = rf_full.predict(Xps); gp = gbdt_full.predict(Xps)
        xp2 = xgb_full.predict(Xps); sp = svr_full.predict(Xps)
        mp = meta.predict(np.column_stack([rp, gp, xp2, sp]))
        times = dd['DATETIME'].dt.strftime('%H:%M').values
        for i in range(len(dd)):
            pred_rows.append({'日期': ds, '时间': times[i],
                'RF': round(max(0, rp[i]), 4), 'GBDT': round(max(0, gp[i]), 4),
                'XGBoost': round(max(0, xp2[i]), 4), 'SVR': round(max(0, sp[i]), 4),
                'Stacking': round(max(0, mp[i]), 4)})
        avg_pred = np.mean([max(0, v) for v in mp])
        print(f'  {ds}: Stacking均值={avg_pred:.4f} NTU')
    pd.DataFrame(pred_rows).to_excel(os.path.join(RESULT_DIR, 'problem1_predictions.xlsx'), index=False)

    # --- 绘图 ---
    # 图1：三层特征筛选
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, d, title, c in zip(axes,
        [mi_df.head(15), imp_df.head(15), perm_df.head(15)],
        ['互信息评分', 'XGBoost特征重要性', '排列重要性'],
        ['#3182CE', '#38A169', '#DD6B20']):
        ax.barh(range(len(d)), d.iloc[:, 1], color=c, height=0.7)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d['Feature'], fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold'); ax.invert_yaxis()
    fig.suptitle('问题1：三层递进特征筛选', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p1_feature_selection.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：模型性能对比
    fig, ax = plt.subplots(figsize=(13, 7))
    nm = list(results.keys()); r2v = [results[n]['R2'] for n in nm]
    rv = [results[n]['RMSE'] for n in nm]
    xp = np.arange(len(nm)); w = 0.35
    colors = ['#3182CE', '#38A169', '#DD6B20', '#805AD5', '#E53E3E']
    ax.bar(xp - w/2, r2v, w, color=colors, alpha=0.85, edgecolor='white', label='R²')
    axt = ax.twinx()
    axt.bar(xp + w/2, rv, w, color=colors, alpha=0.3, edgecolor='white', label='RMSE')
    ax.set_xticks(xp); ax.set_xticklabels(nm, fontsize=10)
    ax.set_title('问题1：模型性能对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p1_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：预测vs实际
    fig, ax = plt.subplots(figsize=(10, 8))
    sn = min(800, len(y_te)); si = np.random.choice(len(y_te), sn, replace=False)
    ax.scatter(y_te[si], stacking_pred[si], alpha=0.5, s=25, c='#3182CE', edgecolors='white', linewidth=0.5)
    mi_v, ma_v = min(y_te.min(), stacking_pred.min()), max(y_te.max(), stacking_pred.max())
    ax.plot([mi_v, ma_v], [mi_v, ma_v], 'r--', linewidth=2)
    ax.set_xlabel('实际值 (NTU)'); ax.set_ylabel('预测值 (NTU)')
    ax.set_title(f'Stacking预测 vs 实际 (R²={results["Stacking"]["R2"]:.4f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p1_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图4：残差分析
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    res = y_te - stacking_pred
    axes[0, 0].scatter(stacking_pred, res, alpha=0.5, s=20, c='#3182CE')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('预测值'); axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差 vs 预测值'); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].hist(res, bins=50, color='#68D391', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_title(f'残差分布 (μ={res.mean():.4f}, σ={res.std():.4f})')
    stats.probplot(res, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot'); axes[1, 0].grid(True, alpha=0.3)
    sn2 = min(150, len(y_te))
    axes[1, 1].plot(range(sn2), y_te[:sn2], 'o-', color='#3182CE', ms=3, lw=1, alpha=0.7, label='实际')
    axes[1, 1].plot(range(sn2), stacking_pred[:sn2], 's-', color='#E53E3E', ms=3, lw=1, alpha=0.7, label='Stacking')
    axes[1, 1].legend(fontsize=8); axes[1, 1].set_title('预测-实际对比'); axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle('问题1：残差分析与检验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p1_residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图5：关键日期预测
    fig, ax = plt.subplots(figsize=(14, 6))
    cols = ['#3182CE', '#38A169', '#DD6B20']
    for idx, ds in enumerate(['2026-02-01', '2026-02-10', '2026-02-20']):
        dp = [r for r in pred_rows if r['日期'] == ds]
        if dp:
            vv = [r['Stacking'] for r in dp]; xv = list(range(len(vv)))
            ax.plot(xv, vv, 'o-', color=cols[idx], lw=2, ms=6, label=ds)
    ax.axhline(y=1.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='国标 ≤1 NTU')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('问题1：关键日期NTU预测', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p1_date_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t_start
    print(f'\n问题1完成! 总耗时: {elapsed:.1f}s')
    print(f'输出: {RESULT_DIR}/')

if __name__ == '__main__':
    main()
