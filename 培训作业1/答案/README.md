# 2020 CUMCM A题：炉温曲线

## 运行

```matlab
>> cd('code'); run_all     % 一键求解 Q1~Q4
>> cd('code'); q1_furnace_curve   % 单独运行
```

## 目录

```
答案/
├── README.md
├── 炉温曲线建模与优化_完整解题过程.md    论文
├── fitted_parameters.mat               拟合参数存档
│
├── code/                               MATLAB 源代码
│   ├── run_all.m                       一键运行
│   ├── solve_heat.m / oven_temp.m      核心求解器
│   ├── analyze_curve.m / check_constraints.m
│   ├── q1_furnace_curve.m              问题1
│   ├── q2_max_speed.m                  问题2
│   ├── q3_min_area.m                   问题3
│   ├── q4_symmetry.m                   问题4
│   └── fit_parameters.m                参数估计
│
└── result/                             输出文件
    └── figures/
        ├── q1_furnace_curve.png
        ├── q2_speed_analysis.png
        ├── q3_optimal.png
        ├── q4_symmetric_optimal.png
        ├── parameter_fit.png
        └── result.csv
```

## 依赖

- MATLAB R2018b+
- 数据文件：`..\题目\附件.xlsx`
