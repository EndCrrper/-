#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三阶段：数据全自动获取与预处理 - A题 自来水厂水质预测与评估"""
import sys, os, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import openpyxl
import xlrd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

COLS_A = {0:'DATE',1:'TIME',2:'RIVERLEVEL',3:'RW_PUMP_DUTY',4:'RW_FLOW',5:'RW_NTU',6:'RW_CLR',7:'RW_PH',8:'FILT_NTU',9:'CW_WELL_LEVEL',10:'CW_PH',11:'CW_NTU',12:'CW_CLR',13:'CL2',14:'F_RIDE',15:'ALUM',16:'TW_PUMP_DUTY',17:'TW_FLOW',18:'T18ML_LEVEL',19:'T18ML_FLOW',20:'REMARKS'}
COLS_B = {0:'DATE',1:'TIME',2:'RIVERLEVEL',4:'RW_FLOW',5:'RW_NTU',6:'RW_CLR',8:'FILT_NTU',9:'CW_WELL_LEVEL',11:'CW_NTU',12:'CW_CLR',14:'TW_FLOW'}
COLS_C = {0:'TIME',1:'RIVERLEVEL',2:'RW_PUMP_DUTY',3:'RW_FLOW',4:'RW_NTU',5:'RW_CLR',6:'RW_PH',7:'FILT_NTU',8:'CW_WELL_LEVEL',9:'CW_PH',10:'CW_NTU',11:'CW_CLR',12:'CL2',13:'F_RIDE',14:'ALUM',15:'TW_PUMP_DUTY',16:'TW_FLOW',17:'T18ML_LEVEL',18:'T18ML_FLOW',19:'REMARKS'}
NUMERIC_COLS = ['RIVERLEVEL','RW_PUMP_DUTY','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','CW_PH','CW_NTU','CW_CLR','CL2','F_RIDE','ALUM','TW_PUMP_DUTY','TW_FLOW','T18ML_LEVEL','T18ML_FLOW']

def excel_serial_to_date(serial):
    if isinstance(serial,(int,float)) and serial > 40000:
        return datetime(1899,12,30)+timedelta(days=int(serial))
    return serial

def parse_time_val(val):
    if val is None or val == '': return ''
    try:
        if isinstance(val,(int,float)):
            val=int(val); h,m=divmod(val,100); return f'{h:02d}:{m:02d}'
        s=str(val).strip()
        if ':' in s: return s
        if len(s)<=4 and s.isdigit():
            h,m=divmod(int(s),100); return f'{h:02d}:{m:02d}'
        return s
    except: return str(val)

def parse_date_val(val):
    if val is None or val == '': return None
    try:
        if isinstance(val,datetime): return val.strftime('%Y-%m-%d')
        if isinstance(val,(int,float)) and val>40000:
            dt=excel_serial_to_date(val)
            if hasattr(dt,'strftime'): return dt.strftime('%Y-%m-%d')
        s=str(val).strip()
        for fmt in ['%Y-%m-%d %H:%M:%S','%Y-%m-%d','%d/%m/%Y','%d-%m-%Y']:
            try: return datetime.strptime(s,fmt).strftime('%Y-%m-%d')
            except: pass
        if ' ' in s: s=s.split(' ')[0]
        return s
    except: return str(val)

def is_header_row(row_vals):
    text=' '.join([str(v) for v in row_vals if v]).upper()
    return any(kw in text for kw in ['DATE','DATA','TIME','RIVER'])

print("="*80)
print("第三阶段：数据全自动获取与预处理")
print("="*80)

# --- Read 2025 ---
base_2025 = r'E:\gitrepo\-\trainingwork0\题目\A题 自来水厂水质预测与评估\附件\附件1  2025数据集'
files_2025 = sorted([f for f in os.listdir(base_2025) if f.endswith('.xlsx')])
all_2025 = []
for f in files_2025:
    wb = openpyxl.load_workbook(os.path.join(base_2025,f))
    ws = wb.active; rows = list(ws.iter_rows(values_only=True))
    if not rows: continue
    col_map = COLS_A if len(rows[0])>=21 else COLS_B
    start_row = 0
    for i,row in enumerate(rows):
        if is_header_row(row): start_row=i+1; break
    for row in rows[start_row:]:
        if row[0] is None: continue
        record={}
        for ci,cn in col_map.items():
            if ci<len(row): record[cn]=row[ci]
            else: record[cn]=None
        record['DATE']=parse_date_val(record.get('DATE'))
        record['TIME']=parse_time_val(record.get('TIME'))
        if record['DATE'] is None: continue
        for nc in NUMERIC_COLS:
            if nc in record and record[nc] is not None:
                try:
                    v=str(record[nc]).replace(',','').replace('-','')
                    record[nc]=np.nan if v=='' or v.upper()=='NONE' else float(v)
                except: record[nc]=np.nan
        for col in ['RW_PUMP_DUTY','TW_PUMP_DUTY']:
            if col in record and record[col] is not None:
                v=str(record[col]).strip()
                if ',' in v or '+' in v:
                    nums=re.findall(r'(\d+)',v)
                    record[col]=float(nums[0]) if nums else np.nan
                else:
                    try: record[col]=float(v)
                    except: record[col]=np.nan
        all_2025.append(record)

df_2025 = pd.DataFrame(all_2025)
print(f"2025年: {len(df_2025)}行")

# --- Read 2026 ---
base_2026 = r'E:\gitrepo\-\trainingwork0\题目\A题 自来水厂水质预测与评估\附件\附件2  2026数据集'
files_2026 = sorted([f for f in os.listdir(base_2026) if f.endswith('.xls')])
all_2026 = []
for f in files_2026:
    m=re.search(r'(\d{4})\u5e74(\d{1,2})\u6708',f)
    if not m: continue
    year,month=int(m.group(1)),int(m.group(2))
    wb=xlrd.open_workbook(os.path.join(base_2026,f))
    for sname in wb.sheet_names():
        dm=re.match(r'(\d{2})\.(\d{2})',sname)
        if not dm: continue
        day=int(dm.group(1)); date_str=f'{year:04d}-{month:02d}-{day:02d}'
        sheet=wb.sheet_by_name(sname)
        if sheet.nrows<2: continue
        for r in range(1,sheet.nrows):
            record={'DATE':date_str}
            for ci,cn in COLS_C.items():
                if ci<sheet.ncols: record[cn]=sheet.cell_value(r,ci)
                else: record[cn]=None
            record['TIME']=parse_time_val(record.get('TIME'))
            for nc in NUMERIC_COLS:
                if nc in record and record[nc] is not None:
                    try:
                        v=str(record[nc]).replace(',','').replace('-','')
                        record[nc]=np.nan if v=='' or v.upper()=='NONE' else float(v)
                    except: record[nc]=np.nan
            for col in ['RW_PUMP_DUTY','TW_PUMP_DUTY']:
                if col in record and record[col] is not None:
                    v=str(record[col]).strip()
                    if ',' in v or '+' in v:
                        nums=re.findall(r'(\d+)',v)
                        record[col]=float(nums[0]) if nums else np.nan
                    else:
                        try: record[col]=float(v)
                        except: record[col]=np.nan
            all_2026.append(record)

df_2026 = pd.DataFrame(all_2026)
print(f"2026年: {len(df_2026)}行")

# --- Merge ---
all_cols = sorted(set(list(df_2025.columns)+list(df_2026.columns)))
for c in all_cols:
    if c not in df_2025.columns: df_2025[c]=np.nan
    if c not in df_2026.columns: df_2026[c]=np.nan
df_all = pd.concat([df_2025[all_cols],df_2026[all_cols]],ignore_index=True)
df_all['DATETIME']=pd.to_datetime(df_all['DATE']+' '+df_all['TIME'].str.extract(r'(\d{2}):?(\d{2})')[0].fillna('00')+':00',format='%Y-%m-%d %H:%M',errors='coerce')
df_all=df_all.sort_values('DATETIME').reset_index(drop=True)
front=['DATETIME','DATE','TIME']
df_all=df_all[front+[c for c in df_all.columns if c not in front]]
print(f"合并: {len(df_all)}行, 时间: {df_all['DATETIME'].min()} ~ {df_all['DATETIME'].max()}")

# --- Quality Report ---
print("\n缺失率统计:")
for col in NUMERIC_COLS:
    if col in df_all.columns:
        rate=df_all[col].isnull().sum()/len(df_all)*100
        bar='#'*int(rate/5) if rate>0 else ''
        print(f"  {col:20s}: {rate:5.1f}% {bar}")

# --- Cleaning ---
df_clean=df_all.copy()
for col in NUMERIC_COLS:
    if col in df_clean.columns and df_clean[col].isnull().sum()>0:
        df_clean[col]=df_clean[col].interpolate(method='linear',limit_direction='both')
        df_clean[col]=df_clean[col].fillna(method='ffill').fillna(method='bfill')

outlier_counts={}
for col in NUMERIC_COLS:
    if col in df_clean.columns:
        Q1=df_clean[col].quantile(0.25); Q3=df_clean[col].quantile(0.75)
        IQR=Q3-Q1; lower=Q1-3*IQR; upper=Q3+3*IQR
        n_out=((df_clean[col]<lower)|(df_clean[col]>upper)).sum()
        if n_out>0:
            df_clean[col]=df_clean[col].clip(lower,upper)
            outlier_counts[col]=n_out
print(f"\n异常值处理: {sum(outlier_counts.values())}个")

for col in ['RW_NTU','FILT_NTU','CW_NTU']:
    if col in df_clean.columns:
        neg=(df_clean[col]<0).sum()
        if neg>0: df_clean[col]=df_clean[col].clip(lower=0)
for col in ['RW_PH','CW_PH']:
    if col in df_clean.columns: df_clean[col]=df_clean[col].clip(0,14)
for col in ['RW_FLOW','TW_FLOW']:
    if col in df_clean.columns: df_clean[col]=df_clean[col].clip(lower=0)

os.makedirs(r'E:\gitrepo\-\trainingwork0\output',exist_ok=True)
df_clean.to_csv(r'E:\gitrepo\-\trainingwork0\output\cleaned_data.csv',index=False,encoding='utf-8-sig')
print(f"\n清洗完成! 已保存: output/cleaned_data.csv ({len(df_clean)}行)")
print(f"前10行预览:\n{df_clean.head(10).to_string(max_colwidth=12)}")
print(f"后5行预览:\n{df_clean.tail(5).to_string(max_colwidth=12)}")
