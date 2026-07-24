%% 2020A 炉温曲线 全题求解
% 依次运行问题1~4
% 注意: Q3和Q4使用PSO, 耗时约1~3分钟

clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));

fprintf('===== 2020 CUMCM A题: 炉温曲线 =====\n\n');

fprintf('[1/4] 问题1: 给定参数炉温曲线\n');
run('q1_furnace_curve.m');

fprintf('\n[2/4] 问题2: 最大传送带速度\n');
run('q2_max_speed.m');

fprintf('\n[3/4] 问题3: 最小化面积\n');
run('q3_min_area.m');

fprintf('\n[4/4] 问题4: 对称性优化\n');
run('q4_symmetry.m');

fprintf('\n===== 全部完成 =====\n');
