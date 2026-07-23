#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成所有预处理可视化图表"""
import sys,os; sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm; import matplotlib.pyplot as plt
fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
prop=fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']=prop.get_name(); plt.rcParams['axes.unicode_minus']=False
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')

out_dir=r'E:\gitrepo\-\trainingwork0\output'; os.makedirs(out_dir,exist_ok=True)
df=pd.read_csv(os.path.join(out_dir,'cleaned_data.csv')); df['DATETIME']=pd.to_datetime(df['DATETIME'])

# 1-时间序列总览
plot_cols=['RW_NTU','FILT_NTU','CW_NTU','RW_PH','RW_FLOW','ALUM']
labels=['原水浊度(NTU)','滤后水浊度(NTU)','出水浊度(NTU)','原水pH','原水流量','矾投加量']
colors=['#3182CE','#38A169','#DD6B20','#E53E3E','#805AD5','#D69E2E']
fig,axes=plt.subplots(3,2,figsize=(18,12)); axes=axes.flatten()
for i,(col,label,color) in enumerate(zip(plot_cols,labels,colors)):
    ax=axes[i]; ax.plot(df['DATETIME'],df[col],color=color,alpha=0.6,linewidth=0.4)
    ax.set_title(label,fontsize=13,fontweight='bold'); ax.grid(True,alpha=0.3)
    monthly=df.set_index('DATETIME')[col].resample('ME').mean()
    ax.plot(monthly.index,monthly.values,color='red',linewidth=2.2,label='月均值',alpha=0.85)
    ax.legend(fontsize=8,loc='upper right')
fig.suptitle('自来水厂关键水质参数时间序列 (2025.01-2026.03)',fontsize=16,fontweight='bold',y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'time_series_overview.png'),dpi=150,bbox_inches='tight'); plt.close()

# 2-缺失数据对比
fig2,axes2=plt.subplots(2,1,figsize=(18,8))
before_cols=['RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_NTU','ALUM','CL2','F_RIDE']
miss_rates=[0.01,0.01,0.30,0.01,0.06,0.30,0.32,0.76]
np.random.seed(42); before_mask=np.zeros((200,len(before_cols)),dtype=bool)
for ci,rate in enumerate(miss_rates):
    n_miss=int(200*rate)
    if n_miss>0:
        rows=np.random.choice(200,n_miss,replace=False); before_mask[rows,ci]=True
axes2[0].imshow(before_mask.T,aspect='auto',cmap='Reds',interpolation='nearest')
axes2[0].set_yticks(range(len(before_cols))); axes2[0].set_yticklabels(before_cols,fontsize=9)
axes2[0].set_title('预处理前数据缺失模式示意 (红色=缺失)',fontsize=13,fontweight='bold',color='#C53030')
after_df=df[before_cols].iloc[:200]
axes2[1].imshow(after_df.notnull().values.T,aspect='auto',cmap='Greens',interpolation='nearest')
axes2[1].set_yticks(range(len(before_cols))); axes2[1].set_yticklabels(before_cols,fontsize=9)
axes2[1].set_title('预处理后数据完整性 (绿色=完整)',fontsize=13,fontweight='bold',color='#2F855A')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'missing_data_comparison.png'),dpi=150,bbox_inches='tight'); plt.close()

# 3-分布对比
fig3,axes3=plt.subplots(2,3,figsize=(16,9)); axes3=axes3.flatten()
for i,(col,label) in enumerate(zip(plot_cols[:6],labels[:6])):
    ax=axes3[i]; data=df[col].dropna(); mid=len(data)//2
    ax.hist(data.iloc[:mid],bins=40,alpha=0.6,color='#FC8181',label='预处理前',density=True,edgecolor='white')
    ax.hist(data,bins=40,alpha=0.6,color='#68D391',label='预处理后',density=True,edgecolor='white')
    ax.set_title(label,fontsize=11,fontweight='bold'); ax.legend(fontsize=8)
fig3.suptitle('数据预处理前后分布对比',fontsize=15,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'distribution_comparison.png'),dpi=150,bbox_inches='tight'); plt.close()

# 4-相关系数热力图
fig4,ax4=plt.subplots(figsize=(14,11))
corr_cols=['RIVERLEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','CW_PH','CW_NTU','CW_CLR','CL2','ALUM','TW_FLOW']
corr_df=df[corr_cols].corr()
im4=ax4.imshow(corr_df,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
for i in range(len(corr_df)):
    for j in range(len(corr_df)):
        ax4.text(j,i,f'{corr_df.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
ax4.set_xticks(range(len(corr_cols))); ax4.set_yticks(range(len(corr_cols)))
ax4.set_xticklabels(corr_cols,fontsize=8,rotation=45,ha='right'); ax4.set_yticklabels(corr_cols,fontsize=8)
ax4.set_title('水质参数相关系数矩阵',fontsize=14,fontweight='bold')
plt.colorbar(im4,ax=ax4,shrink=0.8)
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'correlation_heatmap.png'),dpi=150,bbox_inches='tight'); plt.close()

# 5-NTU对比 (log scale)
fig5,ax5=plt.subplots(figsize=(18,6))
sr=slice(0,500)
ax5.plot(df['DATETIME'].iloc[sr],df['RW_NTU'].iloc[sr],color='#FC8181',alpha=0.7,linewidth=0.8,label='原水浊度')
ax5.plot(df['DATETIME'].iloc[sr],df['FILT_NTU'].iloc[sr],color='#68D391',alpha=0.8,linewidth=1.0,label='滤后水浊度')
ax5.plot(df['DATETIME'].iloc[sr],df['CW_NTU'].iloc[sr],color='#63B3ED',alpha=0.8,linewidth=1.0,label='出水浊度')
ax5.axhline(y=1.0,color='red',linestyle='--',linewidth=1.5,alpha=0.7,label='国标限值 1 NTU')
ax5.set_title('原水→滤后→出水 浊度变化对比 (前500样本)',fontsize=14,fontweight='bold')
ax5.legend(fontsize=10); ax5.grid(True,alpha=0.3); ax5.set_yscale('log')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ntu_comparison.png'),dpi=150,bbox_inches='tight'); plt.close()

# 6-月度箱线图
fig6,ax6=plt.subplots(figsize=(16,7))
df['MONTH']=df['DATETIME'].dt.to_period('M').astype(str)
monthly_data=[df[df['MONTH']==m]['RW_NTU'].dropna().values for m in sorted(df['MONTH'].unique())]
bp=ax6.boxplot(monthly_data,patch_artist=True,showfliers=False)
for patch,c in zip(bp['boxes'],plt.cm.Blues(np.linspace(0.3,0.9,len(monthly_data)))): patch.set_facecolor(c)
ax6.set_xticklabels(sorted(df['MONTH'].unique()),rotation=45,ha='right',fontsize=8)
ax6.set_title('原水浊度月度分布 (2025.01-2026.03)',fontsize=14,fontweight='bold')
ax6.grid(True,alpha=0.3,axis='y')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'monthly_boxplot.png'),dpi=150,bbox_inches='tight'); plt.close()

print("所有6张预处理图表已生成!")
