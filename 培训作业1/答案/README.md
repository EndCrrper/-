# 2020 CUMCM A题：炉温曲线 — 文件清单

## 运行方式

```matlab
>> run_all          % 一键求解 Q1~Q4
>> q1_furnace_curve  % 单独运行某题
>> fit_parameters    % 重新拟合参数（耗时约3分钟）
```

## 文件结构

### 核心求解模块

| 文件 | 功能 |
|------|------|
| `solve_heat.m` | 一维 Crank-Nicolson 热传导求解器 |
| `oven_temp.m` | 炉内环境温度分布（分段线性） |
| `analyze_curve.m` | 炉温曲线指标提取 |
| `check_constraints.m` | 制程界限检查 |

### 问题求解脚本

| 文件 | 问题 | 方法 |
|------|:--:|------|
| `q1_furnace_curve.m` | Q1 给定参数炉温曲线 | CN 数值求解 |
| `q2_max_speed.m` | Q2 最大传送带速度 | 二分搜索 |
| `q3_min_area.m` | Q3 最小化面积 | SelPSO |
| `q4_symmetry.m` | Q4 对称性优化 | SelPSO |

### 参数估计

| 文件 | 功能 |
|------|------|
| `fit_parameters.m` | SelPSO 拟合 10 个热力学参数 |
| `fitted_parameters.mat` | 拟合结果存档（R²=0.9999） |

### 一键运行

| 文件 | 功能 |
|------|------|
| `run_all.m` | 依次运行 Q1~Q4 |

### 输出文件

| 文件 | 来源 |
|------|------|
| `result.csv` | Q1 温度数据（0.5s 间隔） |
| `q1_furnace_curve.png` | Q1 炉温曲线图 |
| `q2_speed_analysis.png` | Q2 四象限速度分析图 |
| `q3_optimal.png` | Q3 最优炉温曲线图 |
| `q4_symmetric_optimal.png` | Q4 对称性优化图 |
| `parameter_fit.png` | 参数拟合对比图 |

### 论文

| 文件 | 内容 |
|------|------|
| `炉温曲线建模与优化_完整解题过程.md` | 完整解题论文 |

## 依赖

- MATLAB R2018b+（需要 `yline` 函数）
- 数据文件：`..\题目\附件.xlsx`（实验数据）
