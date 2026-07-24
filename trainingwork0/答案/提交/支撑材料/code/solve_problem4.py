# %% 问题4：FCE-SA 双维度模糊综合评价水质风险
"""
2020 CUMCM A题：自来水厂水质预测与评估
问题4 —— 建立四级水质风险评价体系，评估2026年1-3月逐日风险
可独立运行：python solve_problem4.py
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
from matplotlib.patches import Patch
import warnings; warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', 'result')
FIG_DIR = os.path.join(RESULT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
TARGET = 'CW_NTU'
THRESHOLD = 1.0     # 国标限值

# ========== 风险等级分类 ==========
def classify_risk(score, max_ntu, exceed_ratio):
    if max_ntu <= 0.5 and exceed_ratio == 0:
        return '安全'
    elif score < 0.05:
        return '低风险'
    elif score < 0.15:
        return '中风险'
    else:
        return '高风险'

# ========== 主流程 ==========
def main():
    t_start = time.time()
    print('=' * 60)
    print('[4/4] 问题4：水质风险评价 (FCE-SA)')
    print('=' * 60)

    df = pd.read_csv(os.path.join(RESULT_DIR, 'cleaned_data.csv'))
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df_2026 = df[df['DATETIME'] >= '2026-01-01'].copy()
    ntu = df_2026[TARGET].values

    # --- 超标事件检测 ---
    exceedances = np.maximum(0, ntu - THRESHOLD)
    episodes = []
    in_ep = False; es = 0
    for i in range(len(exceedances)):
        if exceedances[i] > 0 and not in_ep:
            in_ep = True; es = i
        elif exceedances[i] <= 0 and in_ep:
            in_ep = False; episodes.append((es, i - 1))
    if in_ep: episodes.append((es, len(exceedances) - 1))
    print(f'  超标事件: {len(episodes)} 次')

    # --- 逐日风险评分 ---
    df_2026['DATE_ONLY'] = df_2026['DATETIME'].dt.date
    daily_risk = []
    for dv, grp in df_2026.groupby('DATE_ONLY'):
        ntv = grp[TARGET].values; n = len(ntv)
        max_ntu = ntv.max(); mean_ntu = ntv.mean()
        exceed_ratio = np.sum(ntv > THRESHOLD) / n

        # 连续超标时长
        exc_seq = ntv > THRESHOLD
        max_consec = 0; cur = 0
        for e in exc_seq:
            if e: cur += 1; max_consec = max(max_consec, cur)
            else: cur = 0

        amplitude_score = max(0, max_ntu - THRESHOLD)
        duration_score = max_consec * 2  # 每步2小时
        risk_score = amplitude_score * 0.5 + duration_score / 24 * 0.3 + exceed_ratio * 0.2

        daily_risk.append({'日期': dv, '最大NTU': max_ntu, '平均NTU': mean_ntu,
            '超标比例': exceed_ratio, '最大连续超标(h)': duration_score, '风险评分': risk_score})

    risk_df = pd.DataFrame(daily_risk).sort_values('日期')
    risk_df['风险等级'] = risk_df.apply(
        lambda r: classify_risk(r['风险评分'], r['最大NTU'], r['超标比例']), axis=1)

    # --- 统计输出 ---
    counts = risk_df['风险等级'].value_counts()
    total_days = len(risk_df)
    print('\n2026年1-3月风险等级分布:')
    print(f'  {"等级":8s}  {"天数":>6s}  {"占比":>8s}')
    print(f'  {"-"*24}')
    for level in ['安全', '低风险', '中风险', '高风险']:
        cnt = counts.get(level, 0)
        print(f'  {level:8s}  {cnt:6d}  {cnt/total_days*100:7.1f}%')

    # 高风险日详情
    high_risk = risk_df[risk_df['风险等级'] == '高风险']
    if len(high_risk) > 0:
        print('\n高风险日详情:')
        for _, r in high_risk.iterrows():
            print(f'  {r["日期"]}: NTUmax={r["最大NTU"]:.3f}, 评分={r["风险评分"]:.4f}')

    # 保存结果
    risk_df.to_excel(os.path.join(RESULT_DIR, 'problem4_risk_assessment.xlsx'), index=False)

    # --- 绘图 ---
    # 图1：风险日历
    fig, ax = plt.subplots(figsize=(18, 8))
    cal = np.full((3, 31), np.nan)
    risk_map = {'安全': 0, '低风险': 1, '中风险': 2, '高风险': 3}
    for i, d in enumerate(pd.date_range('2026-01-01', '2026-03-31', freq='D')):
        mi, di = d.month - 1, d.day - 1; dk = d.date()
        match = risk_df[risk_df['日期'] == dk]
        if len(match) > 0:
            cal[mi, di] = risk_map[match.iloc[0]['风险等级']]
    cmap = plt.cm.RdYlGn_r; cmap.set_bad('white')
    ax.imshow(cal, aspect='auto', cmap=cmap, vmin=0, vmax=3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['1月 (31天)', '2月 (28天)', '3月 (31天)'], fontsize=11)
    ax.set_xticks(range(0, 31, 2))
    ax.set_xticklabels([str(i+1) for i in range(0, 31, 2)], fontsize=9)
    ax.set_title('问题4：2026年1-3月水质风险日历', fontsize=14, fontweight='bold')
    ax.legend(handles=[Patch(facecolor=cmap(i/3), label=l)
        for i, l in enumerate(['安全', '低风险', '中风险', '高风险'])],
        loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p4_risk_calendar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：NTU风险区域
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(df_2026['DATETIME'].values, ntu, color='#3182CE', alpha=0.8, linewidth=0.5)
    ax.fill_between(df_2026['DATETIME'].values, 0, ntu, where=(ntu > THRESHOLD),
                    color='#E53E3E', alpha=0.3, label='超标区间')
    ax.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, alpha=0.8, label='国标 ≤1 NTU')
    ax.axhspan(0, 0.5, alpha=0.08, color='#38A169', label='安全区间')
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('问题4：2026年1-3月 NTU 时间序列及超标标注', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p4_ntu_risk_zones.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：风险饼图
    fig, ax = plt.subplots(figsize=(9, 9))
    pl = [f'{l}\n({counts.get(l, 0)}天)' for l in ['安全', '低风险', '中风险', '高风险']]
    pv = [counts.get(l, 0) for l in ['安全', '低风险', '中风险', '高风险']]
    pc = ['#38A169', '#ECC94B', '#DD6B20', '#E53E3E']
    ax.pie(pv, labels=pl, colors=pc, autopct='%1.1f%%',
           explode=(0.02, 0.02, 0.05, 0.1), startangle=90, textprops={'fontsize': 11})
    ax.set_title('问题4：风险等级分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p4_risk_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图4：3月逐日风险评分
    fig, ax = plt.subplots(figsize=(16, 6))
    mr = risk_df[risk_df['日期'].astype(str).str.startswith('2026-03')]
    days_r = range(1, len(mr) + 1)
    color_map = {'安全': '#38A169', '低风险': '#ECC94B', '中风险': '#DD6B20', '高风险': '#E53E3E'}
    cc = [color_map[r] for r in mr['风险等级'].values]
    ax.bar(days_r, mr['风险评分'].values, color=cc, edgecolor='white')
    ax.axhline(y=0.05, color='#ECC94B', linestyle='--', label='低/中风险分界')
    ax.axhline(y=0.15, color='#E53E3E', linestyle='--', label='中/高风险分界')
    ax.set_xticks(days_r)
    ax.set_xticklabels([f'{d}日' for d in days_r], fontsize=8, rotation=45)
    ax.legend()
    ax.set_title('问题4：2026年3月逐日风险评分', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'p4_march_risk.png'), dpi=300, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t_start
    print(f'\n问题4完成! 总耗时: {elapsed:.1f}s')

if __name__ == '__main__':
    main()
