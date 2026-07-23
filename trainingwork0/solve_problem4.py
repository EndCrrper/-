# -*- coding: utf-8 -*-
"""
Problem 4: FCE-SA Risk Assessment
==================================
Use CW_NTU with threshold = 1.0 NTU (national standard)
Find exceedance episodes (consecutive hours > 1.0)
Calculate risk score S = 0.5*A + 0.3*D + 0.2*F
Classify into 4 levels: 安全 / 低风险 / 中风险 / 高风险
Focus on 2026 Jan-Mar
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
from datetime import datetime, timedelta

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
print("Problem 4: FCE-SA Risk Assessment for Water Quality")
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

# Add time components
df['DATE_DT'] = pd.to_datetime(df['DATE'], errors='coerce')
df['HOUR'] = pd.to_datetime(df['TIME'], format='%H:%M', errors='coerce').dt.hour
df['MONTH'] = df['DATE_DT'].dt.month
df['DAY'] = df['DATE_DT'].dt.day
df['WEEKDAY'] = df['DATE_DT'].dt.weekday()  # Monday=0, Sunday=6

# ============================================================
# 3. Risk Assessment Configuration
# ============================================================
THRESHOLD_NTU = 1.0  # National standard for drinking water turbidity
print(f"\n[Config] Risk threshold: CW_NTU > {THRESHOLD_NTU} NTU (National Standard)")

# Focus period: 2026 January to March
focus_start = pd.Timestamp('2026-01-01')
focus_end = pd.Timestamp('2026-03-31 23:00:00')
focus_mask = (df['DATE_DT'] >= focus_start) & (df['DATE_DT'] <= focus_end)
df_focus = df[focus_mask].copy()

print(f"[Focus] Period: 2026-01-01 to 2026-03-31")
print(f"[Focus] Data points: {len(df_focus)}")

if len(df_focus) == 0:
    print("[ERROR] No data in focus period!")
    # Fall back to all available data
    df_focus = df.copy()
    print(f"[Focus] Falling back to all data: {len(df_focus)} points")

# ============================================================
# 4. Find Exceedance Episodes
# ============================================================
print("\n" + "=" * 70)
print("Exceedance Episode Detection")
print("=" * 70)

# Mark exceedances
df_focus['EXCEEDANCE'] = df_focus['CW_NTU'] > THRESHOLD_NTU
n_exceedance = df_focus['EXCEEDANCE'].sum()
n_total = len(df_focus)
exceedance_rate = n_exceedance / n_total * 100
print(f"  Total points: {n_total}")
print(f"  Exceedance points (>1.0 NTU): {n_exceedance} ({exceedance_rate:.2f}%)")

# Find consecutive exceedance episodes
episodes = []
current_episode = []
episode_id = 0

for idx, row in df_focus.iterrows():
    if row['EXCEEDANCE']:
        current_episode.append(idx)
    else:
        if len(current_episode) >= 2:  # Minimum 2 consecutive points for an episode
            episodes.append({
                'Episode_ID': episode_id,
                'Start_Index': current_episode[0],
                'End_Index': current_episode[-1],
                'Start_Time': df_focus.loc[current_episode[0], 'DATETIME'],
                'End_Time': df_focus.loc[current_episode[-1], 'DATETIME'],
                'Duration_Hours': len(current_episode) * 2,  # Each step = 2 hours
                'Peak_NTU': df_focus.loc[current_episode, 'CW_NTU'].max(),
                'Mean_NTU': df_focus.loc[current_episode, 'CW_NTU'].mean(),
                'Max_Exceedance': df_focus.loc[current_episode, 'CW_NTU'].max() - THRESHOLD_NTU
            })
            episode_id += 1
        current_episode = []

# Check last episode
if len(current_episode) >= 2:
    episodes.append({
        'Episode_ID': episode_id,
        'Start_Index': current_episode[0],
        'End_Index': current_episode[-1],
        'Start_Time': df_focus.loc[current_episode[0], 'DATETIME'],
        'End_Time': df_focus.loc[current_episode[-1], 'DATETIME'],
        'Duration_Hours': len(current_episode) * 2,
        'Peak_NTU': df_focus.loc[current_episode, 'CW_NTU'].max(),
        'Mean_NTU': df_focus.loc[current_episode, 'CW_NTU'].mean(),
        'Max_Exceedance': df_focus.loc[current_episode, 'CW_NTU'].max() - THRESHOLD_NTU
    })

episodes_df = pd.DataFrame(episodes) if episodes else pd.DataFrame()
print(f"  Detected exceedance episodes: {len(episodes)}")

if len(episodes) > 0:
    print(f"\n  Top 5 most severe episodes (by peak NTU):")
    for _, ep in episodes_df.nlargest(5, 'Peak_NTU').iterrows():
        print(f"    Ep.{ep['Episode_ID']}: {ep['Start_Time']} -> {ep['End_Time']}, "
              f"Duration={ep['Duration_Hours']}h, Peak={ep['Peak_NTU']:.2f}, Mean={ep['Mean_NTU']:.2f}")

# ============================================================
# 5. FCE-SA Risk Scoring
# ============================================================
print("\n" + "=" * 70)
print("FCE-SA Risk Scoring")
print("  S = 0.5*A + 0.3*D + 0.2*F")
print("  A: Amplitude factor (exceedance magnitude)")
print("  D: Duration factor")
print("  F: Frequency factor")
print("=" * 70)

# Assign risk scores to each hour in the focus period
risk_records = []

# Calculate weekly frequency factor (F)
df_focus['WEEK_NUM'] = df_focus['DATE_DT'].dt.isocalendar().week.astype(int)
weekly_exceedance = df_focus.groupby('WEEK_NUM')['EXCEEDANCE'].mean()
max_weekly_rate = weekly_exceedance.max() if len(weekly_exceedance) > 0 else 1

for idx, row in df_focus.iterrows():
    cw_ntu = row['CW_NTU']

    if not row['EXCEEDANCE']:
        risk_records.append({
            'DATETIME': row['DATETIME'],
            'DATE': row['DATE'],
            'TIME': row['TIME'],
            'CW_NTU': cw_ntu,
            'A_Amplitude': 0,
            'D_Duration': 0,
            'F_Frequency': 0,
            'Risk_Score': 0,
            'Risk_Level': '安全',
            'Risk_Level_EN': 'Safe',
            'Risk_Level_Num': 0
        })
        continue

    # A: Amplitude factor (0-1 scale, how much above threshold)
    # Normalize by max possible (assume max realistic NTU = 5.0)
    exceedance_magnitude = (cw_ntu - THRESHOLD_NTU) / 4.0  # 4.0 = assumed max exceedance
    A = min(max(exceedance_magnitude, 0), 1.0)

    # D: Duration factor (based on episode duration)
    # Find which episode this belongs to
    matched_ep = episodes_df[
        (episodes_df['Start_Index'] <= idx) & (episodes_df['End_Index'] >= idx)
    ]
    if len(matched_ep) > 0:
        duration_hours = matched_ep.iloc[0]['Duration_Hours']
        # Normalize: 48h continuous exceedance = D=1.0
        D = min(duration_hours / 48.0, 1.0)
    else:
        D = 0.02  # Single point exceedance

    # F: Frequency factor (based on weekly exceedance rate)
    week_num = row['WEEK_NUM']
    weekly_rate = weekly_exceedance.get(week_num, 0)
    F = min(weekly_rate, 1.0)  # Proportion of exceedance in the week

    # Risk score
    S = 0.5 * A + 0.3 * D + 0.2 * F

    # Classification
    if S == 0:
        level = '安全'
        level_en = 'Safe'
        level_num = 0
    elif S <= 0.25:
        level = '低风险'
        level_en = 'Low Risk'
        level_num = 1
    elif S <= 0.50:
        level = '中风险'
        level_en = 'Medium Risk'
        level_num = 2
    else:
        level = '高风险'
        level_en = 'High Risk'
        level_num = 3

    risk_records.append({
        'DATETIME': row['DATETIME'],
        'DATE': row['DATE'],
        'TIME': row['TIME'],
        'CW_NTU': cw_ntu,
        'A_Amplitude': round(A, 4),
        'D_Duration': round(D, 4),
        'F_Frequency': round(F, 4),
        'Risk_Score': round(S, 4),
        'Risk_Level': level,
        'Risk_Level_EN': level_en,
        'Risk_Level_Num': level_num
    })

risk_df = pd.DataFrame(risk_records)

# Summary statistics
risk_counts = risk_df['Risk_Level'].value_counts()
print(f"\n[Risk Distribution]")
for level in ['安全', '低风险', '中风险', '高风险']:
    count = risk_counts.get(level, 0)
    pct = count / len(risk_df) * 100
    print(f"  {level}: {count} ({pct:.1f}%)")

print(f"\n  Overall Risk Score: mean={risk_df['Risk_Score'].mean():.4f}, "
      f"max={risk_df['Risk_Score'].max():.4f}")
print(f"  High risk hours: {risk_counts.get('高风险', 0)}")

# ============================================================
# 6. Generate Charts
# ============================================================
print("\n" + "=" * 70)
print("Generating Charts")
print("=" * 70)

# Define risk colors
RISK_COLORS = {'安全': '#2ECC71', '低风险': '#F1C40F', '中风险': '#E67E22', '高风险': '#E74C3C'}
RISK_ORDER = ['安全', '低风险', '中风险', '高风险']

# --- Chart 1: Risk Calendar (heatmap of daily max risk) ---
print("[Chart] Generating risk calendar...")
risk_df['DATE_DT'] = pd.to_datetime(risk_df['DATE'], errors='coerce')
daily_risk = risk_df.groupby('DATE_DT')['Risk_Score'].max().reset_index()
daily_risk.columns = ['Date', 'Max_Risk']
daily_risk['Month'] = daily_risk['Date'].dt.month
daily_risk['Day'] = daily_risk['Date'].dt.day
daily_risk['WeekOfYear'] = daily_risk['Date'].dt.isocalendar().week.astype(int)
daily_risk['Weekday'] = daily_risk['Date'].dt.weekday()

fig, ax = plt.subplots(figsize=(16, 4))

# Create pivot table for heatmap
pivot = daily_risk.pivot_table(
    index='Weekday', columns='WeekOfYear', values='Max_Risk', aggfunc='max'
)
pivot = pivot.reindex(index=[6, 5, 4, 3, 2, 1, 0])  # Sun first (Chinese convention)

if len(pivot) > 0 and not pivot.empty:
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_yticks(range(7))
    ax.set_yticklabels(['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon'], fontsize=9)
    ax.set_xlabel('Week of Year')
    ax.set_title('2026 Q1 Daily Risk Calendar (Max Risk Score per Day)')

    plt.colorbar(im, ax=ax, label='Risk Score', shrink=0.8)
else:
    ax.text(0.5, 0.5, 'Insufficient data for risk calendar', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
cal_path = os.path.join(OUTPUT_DIR, 'p4_risk_calendar.png')
plt.savefig(cal_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {cal_path}")

# --- Chart 2: NTU Time Series with Risk Zones ---
print("[Chart] Generating NTU risk zones chart...")
fig, ax = plt.subplots(figsize=(18, 6))

time_vals = risk_df['DATETIME']
ntu_vals = risk_df['CW_NTU']

# Plot NTU line
ax.plot(time_vals, ntu_vals, 'b-', linewidth=1, alpha=0.7, label='CW_NTU')

# Shade risk zones
for level in reversed(RISK_ORDER):
    mask = risk_df['Risk_Level'] == level
    if mask.any():
        ax.fill_between(
            time_vals[mask], 0, ntu_vals[mask],
            color=RISK_COLORS[level], alpha=0.3, label=level
        )

# Threshold line
ax.axhline(y=THRESHOLD_NTU, color='red', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD_NTU} NTU)')

ax.set_xlabel('Date')
ax.set_ylabel('CW_NTU')
ax.set_title('CW_NTU Time Series with Risk Zones (2026 Q1)')
ax.legend(loc='upper left', ncol=3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
risk_zone_path = os.path.join(OUTPUT_DIR, 'p4_ntu_risk_zones.png')
plt.savefig(risk_zone_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {risk_zone_path}")

# --- Chart 3: Risk Distribution Pie Chart ---
print("[Chart] Generating risk distribution pie...")
fig, ax = plt.subplots(figsize=(8, 8))

risk_dist = risk_df['Risk_Level'].value_counts()
risk_dist = risk_dist.reindex(RISK_ORDER).fillna(0).astype(int)
colors = [RISK_COLORS[l] for l in RISK_ORDER]

wedges, texts, autotexts = ax.pie(
    risk_dist.values,
    labels=RISK_ORDER,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=[0.05, 0.05, 0.05, 0.05]
)
ax.set_title('Risk Level Distribution (2026 Q1)', fontsize=14)

# Add total count
total_text = f'Total hours: {len(risk_df)}'
ax.text(0, -1.3, total_text, ha='center', fontsize=10)

plt.tight_layout()
pie_path = os.path.join(OUTPUT_DIR, 'p4_risk_pie.png')
plt.savefig(pie_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {pie_path}")

# --- Chart 4: March Risk Detail ---
print("[Chart] Generating March risk analysis...")
march_mask = risk_df['DATE_DT'].dt.month == 3
march_df = risk_df[march_mask].copy()

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

if len(march_df) > 0:
    # Top: NTU and risk score in March
    ax1 = axes[0]
    ax1.plot(march_df['DATETIME'], march_df['CW_NTU'], 'b-', linewidth=1.5, label='CW_NTU')
    ax1.fill_between(march_df['DATETIME'], 0, march_df['CW_NTU'],
                      where=(march_df['Risk_Level'] == '高风险'),
                      color='red', alpha=0.3, label='高风险')
    ax1.fill_between(march_df['DATETIME'], 0, march_df['CW_NTU'],
                      where=(march_df['Risk_Level'] == '中风险'),
                      color='orange', alpha=0.3, label='中风险')
    ax1.fill_between(march_df['DATETIME'], 0, march_df['CW_NTU'],
                      where=(march_df['Risk_Level'] == '低风险'),
                      color='yellow', alpha=0.3, label='低风险')
    ax1.axhline(y=THRESHOLD_NTU, color='red', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('CW_NTU')
    ax1.set_title('March 2026: CW_NTU and Risk Zones')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom: Risk score components
    ax2 = axes[1]
    ax2.stackplot(march_df['DATETIME'],
                  march_df['A_Amplitude'] * 0.5,
                  march_df['D_Duration'] * 0.3,
                  march_df['F_Frequency'] * 0.2,
                  labels=['A (Amplitude)', 'D (Duration)', 'F (Frequency)'],
                  colors=['#E74C3C', '#3498DB', '#2ECC71'],
                  alpha=0.7)
    ax2.set_xlabel('Date (March 2026)')
    ax2.set_ylabel('Risk Score Components')
    ax2.set_title('Risk Score Decomposition: S = 0.5*A + 0.3*D + 0.2*F')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'No March data available', ha='center', va='center', transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, 'No March data available', ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
march_path = os.path.join(OUTPUT_DIR, 'p4_march_risk.png')
plt.savefig(march_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[Chart] Saved: {march_path}")

# ============================================================
# 7. Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

# Save full risk assessment
risk_out_cols = ['DATETIME', 'DATE', 'TIME', 'CW_NTU', 'A_Amplitude', 'D_Duration',
                 'F_Frequency', 'Risk_Score', 'Risk_Level']
risk_out = risk_df[risk_out_cols].copy()
risk_out_path = os.path.join(OUTPUT_DIR, 'problem4_risk_assessment.xlsx')
risk_out.to_excel(risk_out_path, index=False, engine='openpyxl')
print(f"[Results] Saved full risk assessment to: {risk_out_path}")

# Save exceedance episodes
if len(episodes) > 0:
    ep_path = os.path.join(OUTPUT_DIR, 'problem4_exceedance_episodes.xlsx')
    episodes_df.to_excel(ep_path, index=False, engine='openpyxl')
    print(f"[Results] Saved exceedance episodes to: {ep_path}")
else:
    print(f"[Results] No exceedance episodes to save.")

# Save risk summary
risk_summary = pd.DataFrame({
    'Risk_Level': ['安全', '低风险', '中风险', '高风险'],
    'Score_Range': ['0', '0-0.25', '0.25-0.5', '>0.5'],
    'Hour_Count': [risk_counts.get('安全', 0), risk_counts.get('低风险', 0),
                   risk_counts.get('中风险', 0), risk_counts.get('高风险', 0)],
    'Percentage': [risk_counts.get('安全', 0) / len(risk_df) * 100,
                   risk_counts.get('低风险', 0) / len(risk_df) * 100,
                   risk_counts.get('中风险', 0) / len(risk_df) * 100,
                   risk_counts.get('高风险', 0) / len(risk_df) * 100]
})
summary_path = os.path.join(OUTPUT_DIR, 'problem4_risk_summary.xlsx')
risk_summary.to_excel(summary_path, index=False, engine='openpyxl')
print(f"[Results] Saved risk summary to: {summary_path}")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY - Problem 4: FCE-SA Risk Assessment")
print("=" * 70)
print(f"  Assessment period: 2026-01-01 to 2026-03-31")
print(f"  Threshold: {THRESHOLD_NTU} NTU")
print(f"  Total hours assessed: {len(risk_df)}")
print(f"  Exceedance hours (>1.0 NTU): {n_exceedance} ({exceedance_rate:.1f}%)")
print(f"  Detected episodes: {len(episodes)}")
print(f"  Risk Score Formula: S = 0.5*A + 0.3*D + 0.2*F")
print(f"  Risk Distribution:")
for level in RISK_ORDER:
    count = risk_counts.get(level, 0)
    pct = count / len(risk_df) * 100
    print(f"    {level}: {count} ({pct:.1f}%)")
print(f"  Mean Risk Score: {risk_df['Risk_Score'].mean():.4f}")
print(f"  Max Risk Score: {risk_df['Risk_Score'].max():.4f}")
if len(episodes) > 0:
    print(f"  Worst episode: Peak={episodes_df['Peak_NTU'].max():.2f} NTU, "
          f"Max Duration={episodes_df['Duration_Hours'].max()}h")
print(f"  Output files saved to: {OUTPUT_DIR}")
print("=" * 70)
print("Problem 4 completed successfully.")
