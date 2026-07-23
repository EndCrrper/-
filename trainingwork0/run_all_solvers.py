#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A题 自来水厂水质预测与评估 - 完整求解脚本（含全部4个问题）"""
import sys,os,math,re,time
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm; import matplotlib.pyplot as plt
fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
prop=fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']=prop.get_name(); plt.rcParams['axes.unicode_minus']=False
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from scipy import signal
import scipy.stats as stats
from itertools import product
import warnings; warnings.filterwarnings('ignore')

out_dir=r'E:\gitrepo\-\trainingwork0\output'; os.makedirs(out_dir,exist_ok=True)
df=pd.read_csv(os.path.join(out_dir,'cleaned_data.csv')); df['DATETIME']=pd.to_datetime(df['DATETIME'])
t0_all=time.time()

# ============================================================
# PROBLEM 2: TD-NARX
# ============================================================
print("\n"+"="*80); print("问题2：滤后水浊度(FILT.NTU)动态时滞建模"); print("="*80)
TARGET2='FILT_NTU'; INPUTS2=['RW_NTU','RW_PH','ALUM','RW_FLOW']
data2=df[['DATETIME']+INPUTS2+[TARGET2]].copy()
for col in INPUTS2+[TARGET2]: data2[col]=data2[col].fillna(method='ffill').fillna(data2[col].median())

# CCF
max_lag_ccf=48; lag_estimates={}
fig_ccf,axes_ccf=plt.subplots(2,2,figsize=(16,10)); axes_ccf=axes_ccf.flatten()
for idx,var in enumerate(INPUTS2):
    x=data2[var].values; y=data2[TARGET2].values
    valid=~(np.isnan(x)|np.isnan(y)); xc,yc=x[valid],y[valid]
    if xc.std()<1e-6 or yc.std()<1e-6: lag_estimates[var]=4; continue
    ccf=signal.correlate(yc-yc.mean(),xc-xc.mean(),mode='full')
    ccf=ccf/(len(xc)*xc.std()*yc.std()+1e-10)
    mid=len(ccf)//2
    neg_filter=np.arange(-mid,len(ccf)-mid)
    usable_mask=(neg_filter<0)&(neg_filter>=-max_lag_ccf)
    usable_lags=-neg_filter[usable_mask]; usable_ccf=np.abs(ccf[usable_mask])
    best_lag=min(usable_lags[np.argmax(usable_ccf)],24) if len(usable_ccf)>0 else 4
    best_corr=usable_ccf[np.argmax(usable_ccf)] if len(usable_ccf)>0 else 0
    lag_estimates[var]=best_lag
    ax=axes_ccf[idx]
    if xc.std()<1e-6 or yc.std()<1e-6:
        ax.text(0.5,0.5,f'{var}\n方差趋近于0\n(默认时滞=4步)',ha='center',va='center',transform=ax.transAxes,fontsize=11,color='gray')
        ax.set_title(f'{var} (无方差)')
    else:
        plot_range=slice(mid-max_lag_ccf,mid+max_lag_ccf); plot_lags=np.arange(-max_lag_ccf,max_lag_ccf)
        ax.plot(plot_lags,ccf[plot_range],color='#3182CE',linewidth=1)
        ax.axvline(x=-best_lag,color='red',linestyle='--',label=f'最优:{best_lag}步({best_lag*2}h)'); ax.axvline(x=0,color='gray',linestyle='-',alpha=0.5)
        ax.legend(fontsize=9)
    ax.set_xlabel('滞后步数'); ax.set_ylabel('CCF'); ax.grid(True,alpha=0.3)
    print(f"  {var}: CCF最优时滞={best_lag}步({best_lag*2}h)")
fig_ccf.suptitle('问题2：CCF互相关时滞估计',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p2_ccf_lag.png'),dpi=150,bbox_inches='tight'); plt.close()

# Grid search
search_ranges={}
for var in INPUTS2:
    base=max(1,min(lag_estimates.get(var,4),12))
    rng=list(range(max(0,base-3),min(max_lag_ccf,base+4)))
    if not rng: rng=[base]
    search_ranges[var]=rng
np.random.seed(42); all_combos=list(product(*search_ranges.values()))
sample_combos=all_combos if len(all_combos)<=200 else [all_combos[i] for i in np.random.choice(len(all_combos),200,replace=False)]
best_score=-np.inf; best_lags=None
for lc in sample_combos:
    X_lag,y_lag=[],[]; mn=max(lc)
    for t in range(mn,len(data2)):
        row=[data2[var].iloc[t-lc[vi]] for vi,var in enumerate(INPUTS2)]
        X_lag.append(row); y_lag.append(data2[TARGET2].iloc[t])
    X_lag=np.array(X_lag); y_lag=np.array(y_lag)
    if len(X_lag)<100: continue
    lr=LinearRegression(); lr.fit(X_lag[:int(len(X_lag)*0.8)],y_lag[:int(len(y_lag)*0.8)])
    score=lr.score(X_lag[int(len(X_lag)*0.8):],y_lag[int(len(y_lag)*0.8):])
    if score>best_score: best_score=score; best_lags=lc
final_lags={var:(best_lags[vi] if best_lags else lag_estimates.get(var,4)) for vi,var in enumerate(INPUTS2)}
print(f"最优异质时滞:"); [print(f"  {var}: {final_lags[var]}步({final_lags[var]*2}h)") for var in INPUTS2]

# NARX
max_lag=max(final_lags.values()); X_narx,y_narx=[],[]
for t in range(max_lag,len(data2)):
    row=[data2[var].iloc[t-final_lags[var]] for var in INPUTS2]
    for al in [1,2,3]:
        if t-al>=0: row.append(data2[TARGET2].iloc[t-al])
    X_narx.append(row); y_narx.append(data2[TARGET2].iloc[t])
X_narx=np.array(X_narx); y_narx=np.array(y_narx)
si2=int(len(X_narx)*0.8); X_tr2,X_te2=X_narx[:si2],X_narx[si2:]; y_tr2,y_te2=y_narx[:si2],y_narx[si2:]
sc2=StandardScaler(); X_tr2_s=sc2.fit_transform(X_tr2); X_te2_s=sc2.transform(X_te2)
narx=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='relu',solver='adam',alpha=0.0001,batch_size=64,learning_rate='adaptive',max_iter=500,early_stopping=True,validation_fraction=0.1,random_state=42)
narx.fit(X_tr2_s,y_tr2); yp2=narx.predict(X_te2_s)
r2_2=r2_score(y_te2,yp2); rmse_2=np.sqrt(mean_squared_error(y_te2,yp2))
print(f"TD-NARX: R²={r2_2:.4f}, RMSE={rmse_2:.4f}")

# Problem 2 charts
fig1,ax1=plt.subplots(figsize=(12,6))
vp=list(final_lags.keys()); lp=[final_lags[v]*2 for v in vp]
bars=ax1.barh(vp,lp,color=['#3182CE','#38A169','#DD6B20','#805AD5'],height=0.5)
[ax1.text(b.get_width()+0.2,b.get_y()+b.get_height()/2,f'{h}h',va='center',fontsize=11,fontweight='bold') for b,h in zip(bars,lp)]
ax1.set_xlabel('最优时滞(小时)'); ax1.set_title('问题2：异质时滞估计',fontsize=14,fontweight='bold'); ax1.grid(True,alpha=0.3,axis='x')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p2_optimal_lags.png'),dpi=150,bbox_inches='tight'); plt.close()

fig2,ax2=plt.subplots(figsize=(14,6)); sn2=min(200,len(y_te2))
ax2.plot(range(sn2),y_te2[:sn2],'o-',color='#3182CE',ms=4,lw=1.5,alpha=0.8,label='实际')
ax2.plot(range(sn2),yp2[:sn2],'s-',color='#E53E3E',ms=4,lw=1.5,alpha=0.8,label=f'TD-NARX(R²={r2_2:.3f})')
ax2.fill_between(range(sn2),yp2[:sn2]-rmse_2,yp2[:sn2]+rmse_2,alpha=0.15,color='#E53E3E',label=f'±RMSE({rmse_2:.3f})')
ax2.legend(); ax2.grid(True,alpha=0.3); ax2.set_title('问题2：TD-NARX预测效果',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p2_narx_prediction.png'),dpi=150,bbox_inches='tight'); plt.close()

fig3,axes3=plt.subplots(1,2,figsize=(14,6))
axes3[0].scatter(y_te2,yp2,alpha=0.5,s=20,c='#3182CE',edgecolors='white',linewidth=0.5)
mv2,mx2=min(y_te2.min(),yp2.min()),max(y_te2.max(),yp2.max())
axes3[0].plot([mv2,mx2],[mv2,mx2],'r--',linewidth=2); axes3[0].set_xlabel('实际'); axes3[0].set_ylabel('预测')
axes3[0].set_title(f'预测vs实际(R²={r2_2:.4f})'); axes3[0].grid(True,alpha=0.3)
res2=y_te2-yp2; axes3[1].hist(res2,bins=50,color='#68D391',edgecolor='white',alpha=0.8)
axes3[1].axvline(x=0,color='r',linestyle='--'); axes3[1].set_title(f'残差(μ={res2.mean():.4f},σ={res2.std():.4f})'); axes3[1].grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p2_fit_accuracy.png'),dpi=150,bbox_inches='tight'); plt.close()

pd.DataFrame([{'变量':v,'时滞(步)':final_lags[v],'时滞(h)':final_lags[v]*2} for v in INPUTS2]).to_csv(os.path.join(out_dir,'problem2_lag_parameters.csv'),index=False)
print("问题2完成!")

# ============================================================
# PROBLEM 3: RTD-GBDT
# ============================================================
print("\n"+"="*80); print("问题3：RTD-GBDT混合动态预测"); print("="*80)
TARGET3='CW_NTU'; TAU=4.0; N_TANKS=3; WINDOW=12
def rtd_weights(tau=TAU,n=N_TANKS,H=WINDOW):
    t=np.arange(1,H+1)
    w=(n/tau)**n*t**(n-1)*np.exp(-n*t/tau)/math.factorial(n-1)
    return w/w.sum()
rtd_w=rtd_weights()

data3=df.copy()
for col in ['FILT_NTU','RW_NTU','RW_FLOW','ALUM']:
    if col in data3.columns:
        data3[col]=data3[col].fillna(method='ffill')
        conv=np.convolve(data3[col].values,rtd_w,mode='same')
        data3[f'{col}_RTD']=conv

data3['HOUR']=data3['DATETIME'].dt.hour; data3['MONTH']=data3['DATETIME'].dt.month
for lag in [1,2,3]:
    data3[f'TARGET_LAG{lag}']=data3[TARGET3].shift(lag)
    for col in ['FILT_NTU','RW_NTU']:
        if col in data3.columns: data3[f'{col}_LAG{lag}']=data3[col].shift(lag)
for col in data3.columns:
    if data3[col].isnull().any(): data3[col]=data3[col].fillna(method='ffill').fillna(0)

feats3=['FILT_NTU_RTD','RW_NTU_RTD','RW_FLOW_RTD','ALUM_RTD','FILT_NTU','RW_NTU','HOUR','MONTH','TARGET_LAG1','TARGET_LAG2','TARGET_LAG3','FILT_NTU_LAG1','FILT_NTU_LAG2','RW_NTU_LAG1','RW_NTU_LAG2']
feats3=[f for f in feats3 if f in data3.columns]
md3=data3[feats3+[TARGET3,'DATETIME']].dropna()
X3=md3[feats3].values; y3=md3[TARGET3].values
ni3=int(len(X3)*0.8); X_tr3,X_te3=X3[:ni3],X3[ni3:]; y_tr3,y_te3=y3[:ni3],y3[ni3:]
sc3=StandardScaler(); X_tr3_s=sc3.fit_transform(X_tr3); X_te3_s=sc3.transform(X_te3)
gbdt3=GradientBoostingRegressor(n_estimators=300,max_depth=8,learning_rate=0.03,subsample=0.8,random_state=42)
gbdt3.fit(X_tr3_s,y_tr3); yp3=gbdt3.predict(X_te3_s)
r2_3=r2_score(y_te3,yp3); rmse_3=np.sqrt(mean_squared_error(y_te3,yp3))
print(f"RTD-GBDT: R²={r2_3:.4f}, RMSE={rmse_3:.4f}")

# Predict target dates
pred_rows3=[]
for date_str in ['2026-02-01','2026-02-10','2026-02-20']:
    dt=pd.to_datetime(date_str)
    dd=md3[(md3['DATETIME']>=dt-pd.Timedelta(days=2))&(md3['DATETIME']<=dt+pd.Timedelta(days=2))]
    if len(dd)==0: continue
    for hour in range(7,20):
        hd=dd[dd['DATETIME'].dt.hour==hour]
        if len(hd)==0: hd=dd.iloc[[0]]
        row=hd.iloc[0]; Xp=row[feats3].values.reshape(1,-1)
        pv=max(0,gbdt3.predict(sc3.transform(Xp))[0])
        av=row[TARGET3] if TARGET3 in row.index else None
        pred_rows3.append({'日期':date_str,'时间':f'{hour:02d}:00','RTD-GBDT预测_NTU':round(pv,4),'实际值_NTU':round(av,4) if av is not None and not np.isnan(av) else 'N/A'})
pd.DataFrame(pred_rows3).to_excel(os.path.join(out_dir,'problem3_predictions.xlsx'),index=False)

# Sensitivity (OAT)
si=ni3+len(y_te3)//2; Xb=X_te3_s[len(y_te3)//2:len(y_te3)//2+1].copy()
sens={}
for i,f in enumerate(feats3[:8]):
    Xp2=Xb.copy(); Xp2[0,i]=Xb[0,i]*1.1
    pb=gbdt3.predict(Xb)[0]; pp=gbdt3.predict(Xp2)[0]
    sens[f]=abs(pp-pb)/(abs(pb)+1e-6)
for f,s in sorted(sens.items(),key=lambda x:-x[1])[:6]: print(f"  敏感性 {f}: {s:.4f}")

# Problem 3 charts
fig31,ax31=plt.subplots(figsize=(10,5))
tv=np.arange(1,25); wv=rtd_weights(tau=TAU,n=N_TANKS,H=24)
ax31.fill_between(tv,wv,alpha=0.5,color='#3182CE'); ax31.plot(tv,wv,'o-',color='#3182CE',lw=2,ms=6)
ax31.axvline(x=TAU,color='red',linestyle='--',label=f'τ={TAU}h'); ax31.legend(); ax31.grid(True,alpha=0.3)
ax31.set_title('清水池停留时间分布(RTD)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p3_rtd_curve.png'),dpi=150,bbox_inches='tight'); plt.close()

fig32,ax32=plt.subplots(figsize=(14,6)); cols3=['#3182CE','#38A169','#DD6B20']
for idx,ds in enumerate(['2026-02-01','2026-02-10','2026-02-20']):
    dp=[r for r in pred_rows3 if r['日期']==ds]
    if dp:
        hh=list(range(7,20)); vv=[r['RTD-GBDT预测_NTU'] for r in dp]
        ax32.plot(hh,vv,'o-',color=cols3[idx],lw=2,ms=6,label=ds)
ax32.axhline(y=1.0,color='red',linestyle='--',lw=1.5,alpha=0.7,label='国标≤1NTU')
ax32.set_xticks(range(7,20)); ax32.legend(ncol=2); ax32.grid(True,alpha=0.3)
ax32.set_title('问题3：2026年2月关键日期7:00-19:00 NTU预测',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p3_date_predictions.png'),dpi=150,bbox_inches='tight'); plt.close()

fig33,ax33=plt.subplots(figsize=(10,6))
fn=[s[0] for s in sorted(sens.items(),key=lambda x:-x[1])[:6]]
fv=[s[1] for s in sorted(sens.items(),key=lambda x:-x[1])[:6]]
ax33.barh(fn,fv,color=plt.cm.Blues(np.linspace(0.4,0.95,len(fn))),edgecolor='white')
ax33.set_xlabel('敏感性'); ax33.set_title('问题3：OAT敏感性分析',fontsize=14,fontweight='bold'); ax33.invert_yaxis(); ax33.grid(True,alpha=0.3,axis='x')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p3_sensitivity.png'),dpi=150,bbox_inches='tight'); plt.close()

fig34,ax34=plt.subplots(figsize=(14,6)); dn=min(300,len(y_te3))
ax34.plot(range(dn),y_te3[:dn],'-',color='#3182CE',lw=1.5,alpha=0.8,label='实际')
ax34.plot(range(dn),yp3[:dn],'-',color='#E53E3E',lw=1.5,alpha=0.8,label=f'RTD-GBDT(R²={r2_3:.4f})')
ax34.fill_between(range(dn),yp3[:dn]-rmse_3,yp3[:dn]+rmse_3,alpha=0.15,color='#E53E3E')
ax34.axhline(y=1.0,color='red',linestyle='--',lw=1,alpha=0.5,label='国标≤1NTU')
ax34.legend(ncol=2); ax34.grid(True,alpha=0.3); ax34.set_title(f'问题3：RTD-GBDT预测效果',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p3_prediction_timeseries.png'),dpi=150,bbox_inches='tight'); plt.close()
print("问题3完成!")

# ============================================================
# PROBLEM 4: FCE-SA
# ============================================================
print("\n"+"="*80); print("问题4：水质风险评价体系"); print("="*80)
TARGET4='CW_NTU'; THRESH=1.0
df4=df[df['DATETIME']>='2026-01-01'].copy()
ntu4=df4[TARGET4].values
exc=np.maximum(0,ntu4-THRESH)

episodes=[]; in_ep=False; es=0
for i in range(len(exc)):
    if exc[i]>0 and not in_ep: in_ep=True; es=i
    elif exc[i]<=0 and in_ep: in_ep=False; episodes.append((es,i-1))
if in_ep: episodes.append((es,len(exc)-1))
print(f"超标事件: {len(episodes)}次")

df4['DATE_ONLY']=df4['DATETIME'].dt.date; daily_risk=[]
for dv,grp in df4.groupby('DATE_ONLY'):
    ntv=grp[TARGET4].values; n=len(ntv)
    max_ntu=ntv.max(); mean_ntu=ntv.mean(); exceed_ratio=np.sum(ntv>THRESH)/n
    exc_seq=ntv>THRESH; max_consec=0; cur=0
    for e in exc_seq:
        if e: cur+=1; max_consec=max(max_consec,cur)
        else: cur=0
    amplitude_score=max(0,max_ntu-THRESH); duration_score=max_consec*2
    risk_score=amplitude_score*0.5+duration_score/24*0.3+exceed_ratio*0.2
    daily_risk.append({'日期':dv,'最大NTU':max_ntu,'平均NTU':mean_ntu,'超标比例':exceed_ratio,'最大连续超标(h)':duration_score,'风险评分':risk_score})

risk_df=pd.DataFrame(daily_risk).sort_values('日期')
def classify(score,mx,er):
    if mx<=0.5 and er==0: return '安全'
    elif score<0.05: return '低风险'
    elif score<0.15: return '中风险'
    else: return '高风险'
risk_df['风险等级']=risk_df.apply(lambda r:classify(r['风险评分'],r['最大NTU'],r['超标比例']),axis=1)

lc=risk_df['风险等级'].value_counts(); td=len(risk_df)
for level in ['安全','低风险','中风险','高风险']:
    cnt=lc.get(level,0); print(f"  {level}: {cnt}天 ({cnt/td*100:.1f}%)")

risk_df.to_excel(os.path.join(out_dir,'problem4_risk_assessment.xlsx'),index=False)

# Problem 4 charts
fig41,ax41=plt.subplots(figsize=(18,8))
cal=np.full((3,31),np.nan)
for i,d in enumerate(pd.date_range('2026-01-01','2026-03-31',freq='D')):
    mi,di=d.month-1,d.day-1; dk=d.date()
    match=risk_df[risk_df['日期']==dk]
    if len(match)>0:
        lm={'安全':0,'低风险':1,'中风险':2,'高风险':3}
        cal[mi,di]=lm[match.iloc[0]['风险等级']]
cmap=plt.cm.RdYlGn_r; cmap.set_bad('white')
ax41.imshow(cal,aspect='auto',cmap=cmap,vmin=0,vmax=3)
ax41.set_yticks([0,1,2]); ax41.set_yticklabels(['1月(31天)','2月(28天)','3月(31天)'],fontsize=11)
ax41.set_xticks(range(0,31,2)); ax41.set_xticklabels([str(i+1) for i in range(0,31,2)],fontsize=9)
ax41.set_title('问题4：2026年1-3月水质风险日历',fontsize=14,fontweight='bold')
from matplotlib.patches import Patch
ax41.legend(handles=[Patch(facecolor=cmap(i/3),label=l) for i,l in enumerate(['安全','低风险','中风险','高风险'])],loc='upper right',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p4_risk_calendar.png'),dpi=150,bbox_inches='tight'); plt.close()

fig42,ax42=plt.subplots(figsize=(18,7))
ax42.plot(df4['DATETIME'].values,ntu4,color='#3182CE',alpha=0.8,linewidth=0.5)
ax42.fill_between(df4['DATETIME'].values,0,ntu4,where=(ntu4>THRESH),color='#E53E3E',alpha=0.3,label='超标区间')
ax42.axhline(y=THRESH,color='red',linestyle='--',linewidth=2,alpha=0.8,label='国标≤1NTU')
ax42.axhspan(0,0.5,alpha=0.08,color='#38A169',label='安全区间')
ax42.legend(ncol=2,fontsize=9); ax42.grid(True,alpha=0.3); ax42.set_title('问题4：2026年1-3月NTU及超标标注',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p4_ntu_risk_zones.png'),dpi=150,bbox_inches='tight'); plt.close()

fig43,ax43=plt.subplots(figsize=(9,9))
pl=[f'{l}\n({lc.get(l,0)}天)' for l in ['安全','低风险','中风险','高风险']]
pv2=[lc.get(l,0) for l in ['安全','低风险','中风险','高风险']]
pc=['#38A169','#ECC94B','#DD6B20','#E53E3E']
ax43.pie(pv2,labels=pl,colors=pc,autopct='%1.1f%%',explode=(0.02,0.02,0.05,0.1),startangle=90,textprops={'fontsize':11})
ax43.set_title('问题4：风险等级分布',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p4_risk_pie.png'),dpi=150,bbox_inches='tight'); plt.close()

fig44,ax44=plt.subplots(figsize=(16,6))
mr=risk_df[risk_df['日期'].astype(str).str.startswith('2026-03')]
days_r=range(1,len(mr)+1)
cc4=[{'安全':'#38A169','低风险':'#ECC94B','中风险':'#DD6B20','高风险':'#E53E3E'}[r] for r in mr['风险等级'].values]
ax44.bar(days_r,mr['风险评分'].values,color=cc4,edgecolor='white')
ax44.axhline(y=0.05,color='#ECC94B',linestyle='--',label='低/中风险界'); ax44.axhline(y=0.15,color='#E53E3E',linestyle='--',label='中/高风险界')
ax44.set_xticks(days_r); ax44.set_xticklabels([f'{d}日' for d in days_r],fontsize=8,rotation=45); ax44.legend()
ax44.set_title('问题4：2026年3月逐日风险评分',fontsize=14,fontweight='bold'); ax44.grid(True,alpha=0.3,axis='y')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p4_march_risk.png'),dpi=150,bbox_inches='tight'); plt.close()
print("问题4完成!")

# ============================================================
# PROBLEM 1: MIFS-XGBoost-Stacking (run last as it takes longest)
# ============================================================
print("\n"+"="*80); print("问题1：因素筛选与NTU预测"); print("="*80)
TARGET1='CW_NTU'
df1=df.copy()
df1['HOUR']=df1['DATETIME'].dt.hour; df1['MONTH']=df1['DATETIME'].dt.month; df1['DAY']=df1['DATETIME'].dt.day; df1['DOW']=df1['DATETIME'].dt.dayofweek
for col in ['RW_NTU','RW_CLR','RW_FLOW','FILT_NTU','ALUM']:
    for w in [6,12,24]: df1[f'{col}_ROLL{w}']=df1[col].rolling(window=w,min_periods=1).mean()
for col in ['RW_NTU','FILT_NTU','RW_FLOW','ALUM']:
    for lag in [1,2]: df1[f'{col}_LAG{lag}']=df1[col].shift(lag)
for col in df1.columns:
    if df1[col].isnull().any(): df1[col]=df1[col].fillna(method='ffill').fillna(df1[col].median() if str(df1[col].dtype) in ['float64','int64'] else 0)
df1=df1.dropna(subset=[TARGET1])
bf=['RIVERLEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','CW_WELL_LEVEL','CW_PH','CL2','ALUM','TW_FLOW','FILT_NTU','CW_CLR','F_RIDE']
af=[f for f in bf+['HOUR','MONTH','DAY','DOW']+[c for c in df1.columns if '_ROLL' in c or '_LAG' in c] if f in df1.columns]
data1=df1[af+[TARGET1,'DATETIME']].dropna(subset=[TARGET1])
for col in af:
    if data1[col].isnull().sum()>0: data1[col]=data1[col].fillna(data1[col].median())
X1=data1[af].values; y1=data1[TARGET1].values

# Three-layer selection
mi_s=mutual_info_regression(X1,y1,random_state=42)
mi_df=pd.DataFrame({'Feature':af,'MI':mi_s}).sort_values('MI',ascending=False)
sel_mi=[f for f,s in zip(af,mi_s) if s>np.median(mi_s)*0.3]
xgb1=XGBRegressor(n_estimators=300,max_depth=8,learning_rate=0.05,random_state=42).fit(X1,y1)
xgb_imp=pd.DataFrame({'Feature':af,'Importance':xgb1.feature_importances_}).sort_values('Importance',ascending=False)
top_xgb=xgb_imp[xgb_imp['Importance'].cumsum()<=0.92]['Feature'].tolist()
if len(top_xgb)<8: top_xgb=xgb_imp.head(12)['Feature'].tolist()
perm_imp=permutation_importance(xgb1,X1,y1,n_repeats=5,random_state=42,n_jobs=-1)
perm_df=pd.DataFrame({'Feature':af,'Importance':perm_imp.importances_mean}).sort_values('Importance',ascending=False)
final_f=list(set(sel_mi)&set(top_xgb)&set(perm_df.head(15)['Feature'].tolist()))
for f in set(sel_mi)&set(top_xgb):
    if f not in final_f: final_f.append(f)
for f in ['RW_NTU','FILT_NTU','ALUM','RW_PH','RW_FLOW']:
    if f in af and f not in final_f: final_f.append(f)
print(f"选定特征: {len(final_f)}个")

# Stacking
Xf=data1[final_f].values; ni1=int(len(Xf)*0.8); X_tr1,X_te1=Xf[:ni1],Xf[ni1:]; y_tr1,y_te1=y1[:ni1],y1[ni1:]
sc1=StandardScaler(); X_tr1_s=sc1.fit_transform(X_tr1); X_te1_s=sc1.transform(X_te1)
kf1=KFold(n_splits=5,shuffle=True,random_state=42)
def oof_pred(model,X_tr,y_tr,X_te):
    oof=np.zeros(len(X_tr)); te=np.zeros(len(X_te))
    for ti,vi in kf1.split(X_tr):
        model.fit(X_tr[ti],y_tr[ti]); oof[vi]=model.predict(X_tr[vi]); te+=model.predict(X_te)/kf1.n_splits
    return oof,te
rf_oof,rf_te=oof_pred(RandomForestRegressor(n_estimators=300,max_depth=12,random_state=42,n_jobs=-1),X_tr1_s,y_tr1,X_te1_s)
gbdt_oof,gbdt_te=oof_pred(GradientBoostingRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,random_state=42),X_tr1_s,y_tr1,X_te1_s)
xgb_oof,xgb_te=oof_pred(XGBRegressor(n_estimators=300,max_depth=8,learning_rate=0.05,random_state=42),X_tr1_s,y_tr1,X_te1_s)
svr_oof,svr_te=oof_pred(SVR(kernel='rbf',C=5,gamma='scale',epsilon=0.01),X_tr1_s,y_tr1,X_te1_s)
meta1=Ridge(alpha=0.5).fit(np.column_stack([rf_oof,gbdt_oof,xgb_oof,svr_oof]),y_tr1)
stacking_pred1=meta1.predict(np.column_stack([rf_te,gbdt_te,xgb_te,svr_te]))
rf_f=RandomForestRegressor(n_estimators=300,max_depth=12,random_state=42,n_jobs=-1).fit(X_tr1_s,y_tr1)
gbdt_f=GradientBoostingRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,random_state=42).fit(X_tr1_s,y_tr1)
xgb_f=XGBRegressor(n_estimators=300,max_depth=8,learning_rate=0.05,random_state=42).fit(X_tr1_s,y_tr1)
svr_f=SVR(kernel='rbf',C=5,gamma='scale',epsilon=0.01).fit(X_tr1_s,y_tr1)
models1={'RF':(rf_f.predict(X_te1_s),'#3182CE'),'GBDT':(gbdt_f.predict(X_te1_s),'#38A169'),'XGBoost':(xgb_f.predict(X_te1_s),'#DD6B20'),'SVR':(svr_f.predict(X_te1_s),'#805AD5'),'Stacking':(stacking_pred1,'#E53E3E')}
results1={}
for nm,(p,_) in models1.items():
    results1[nm]={'R2':r2_score(y_te1,p),'RMSE':np.sqrt(mean_squared_error(y_te1,p)),'MAE':mean_absolute_error(y_te1,p)}
    print(f"  {nm}: R²={results1[nm]['R2']:.4f}, RMSE={results1[nm]['RMSE']:.4f}")

# Predict dates
has_date='DATE' in data1.columns
pred_rows1=[]
for ds in ['2026-02-01','2026-02-10','2026-02-20']:
    if has_date:
        dd=data1[data1['DATE']==ds].copy()
        if len(dd)==0:
            dt=pd.to_datetime(ds); dd=data1[(data1['DATETIME']>=dt-pd.Timedelta(days=3))&(data1['DATETIME']<=dt+pd.Timedelta(days=3))].copy()
    else:
        dt=pd.to_datetime(ds); dd=data1[(data1['DATETIME']>=dt-pd.Timedelta(days=3))&(data1['DATETIME']<=dt+pd.Timedelta(days=3))].copy()
    if len(dd)==0: continue
    Xp=dd[final_f].values; Xps=sc1.transform(Xp)
    rp=rf_f.predict(Xps); gp=gbdt_f.predict(Xps); xp2=xgb_f.predict(Xps); sp=svr_f.predict(Xps)
    mp=meta1.predict(np.column_stack([rp,gp,xp2,sp]))
    tc=dd['TIME'].values if 'TIME' in dd.columns else dd['DATETIME'].dt.strftime('%H:%M').values
    for i in range(len(dd)):
        pred_rows1.append({'日期':ds,'时间':tc[i],'RF':round(max(0,rp[i]),4),'GBDT':round(max(0,gp[i]),4),'XGBoost':round(max(0,xp2[i]),4),'SVR':round(max(0,sp[i]),4),'Stacking':round(max(0,mp[i]),4)})
pd.DataFrame(pred_rows1).to_excel(os.path.join(out_dir,'problem1_predictions.xlsx'),index=False)

# Problem 1 charts
fig51,axes51=plt.subplots(1,3,figsize=(18,7))
for ai,(d,title,c) in enumerate(zip([mi_df.head(15),xgb_imp.head(15),perm_df.head(15)],['MI评分','XGBoost重要性','排列重要性'],['#3182CE','#38A169','#DD6B20'])):
    axes51[ai].barh(range(len(d)),d.iloc[:,1],color=c,height=0.7)
    axes51[ai].set_yticks(range(len(d))); axes51[ai].set_yticklabels(d['Feature'],fontsize=8)
    axes51[ai].set_title(title,fontsize=12,fontweight='bold'); axes51[ai].invert_yaxis()
fig51.suptitle('问题1：三层递进特征筛选',fontsize=14,fontweight='bold',y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p1_feature_selection.png'),dpi=150,bbox_inches='tight'); plt.close()

fig52,ax52=plt.subplots(figsize=(13,7))
nms=list(results1.keys()); r2v=[results1[n]['R2'] for n in nms]; rv=[results1[n]['RMSE'] for n in nms]
xpos=np.arange(len(nms)); bc=['#3182CE','#38A169','#DD6B20','#805AD5','#E53E3E']
ax52.bar(xpos-w/2,r2v,w,color=bc,alpha=0.85,edgecolor='white',label='R²')
axt=ax52.twinx(); axt.bar(xpos+w/2,rv,w,color=bc,alpha=0.3,edgecolor='white',label='RMSE')
ax52.set_xticks(xpos); ax52.set_xticklabels(nms,fontsize=10); ax52.set_title('问题1：模型性能对比',fontsize=14,fontweight='bold'); ax52.legend(loc='upper left')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p1_model_comparison.png'),dpi=150,bbox_inches='tight'); plt.close()

fig53,ax53=plt.subplots(figsize=(10,8))
sn1=min(800,len(y_te1)); si1=np.random.choice(len(y_te1),sn1,replace=False)
ax53.scatter(y_te1[si1],stacking_pred1[si1],alpha=0.5,s=25,c='#3182CE',edgecolors='white',linewidth=0.5)
mv1,mx1=min(y_te1.min(),stacking_pred1.min()),max(y_te1.max(),stacking_pred1.max())
ax53.plot([mv1,mx1],[mv1,mx1],'r--',linewidth=2); ax53.set_xlabel('实际'); ax53.set_ylabel('预测')
ax53.set_title(f'Stacking预测vs实际(R²={results1["Stacking"]["R2"]:.4f})',fontsize=14,fontweight='bold'); ax53.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p1_pred_vs_actual.png'),dpi=150,bbox_inches='tight'); plt.close()

fig54,axes54=plt.subplots(2,2,figsize=(14,10))
res1=y_te1-stacking_pred1
axes54[0,0].scatter(stacking_pred1,res1,alpha=0.5,s=20,c='#3182CE'); axes54[0,0].axhline(y=0,color='r',linestyle='--')
axes54[0,0].set_xlabel('预测值'); axes54[0,0].set_ylabel('残差'); axes54[0,0].set_title('残差vs预测值'); axes54[0,0].grid(True,alpha=0.3)
axes54[0,1].hist(res1,bins=50,color='#68D391',edgecolor='white',alpha=0.8); axes54[0,1].axvline(x=0,color='r',linestyle='--')
axes54[0,1].set_title(f'残差分布(μ={res1.mean():.4f},σ={res1.std():.4f})')
stats.probplot(res1,dist="norm",plot=axes54[1,0]); axes54[1,0].set_title('Q-Q Plot'); axes54[1,0].grid(True,alpha=0.3)
sn12=min(150,len(y_te1))
axes54[1,1].plot(range(sn12),y_te1[:sn12],'o-',color='#3182CE',ms=3,lw=1,alpha=0.7,label='实际')
axes54[1,1].plot(range(sn12),stacking_pred1[:sn12],'s-',color='#E53E3E',ms=3,lw=1,alpha=0.7,label='Stacking')
axes54[1,1].legend(fontsize=8); axes54[1,1].set_title('预测-实际对比'); axes54[1,1].grid(True,alpha=0.3)
fig54.suptitle('问题1：残差分析与检验',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p1_residual_analysis.png'),dpi=150,bbox_inches='tight'); plt.close()

fig55,ax55=plt.subplots(figsize=(14,6))
for idx,ds in enumerate(['2026-02-01','2026-02-10','2026-02-20']):
    dp=[r for r in pred_rows1 if r['日期']==ds]
    if dp:
        vv=[r['Stacking'] for r in dp]; xv=list(range(len(vv)))
        ax55.plot(xv,vv,'o-',color=['#3182CE','#38A169','#DD6B20'][idx],lw=2,ms=6,label=ds)
ax55.axhline(y=1.0,color='red',linestyle='--',lw=1.5,alpha=0.7,label='国标≤1NTU')
ax55.legend(); ax55.grid(True,alpha=0.3); ax55.set_title('问题1：关键日期NTU预测',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'p1_date_predictions.png'),dpi=150,bbox_inches='tight'); plt.close()
print("问题1完成!")

print(f"\n{'='*80}")
print(f"全部求解完成! 总耗时: {time.time()-t0_all:.1f}s")
print(f"输出: {out_dir}")
print("="*80)
