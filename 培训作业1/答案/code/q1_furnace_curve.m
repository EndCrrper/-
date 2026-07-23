%% 问题1: 给定参数的炉温曲线
% v=78 cm/min, T=[173,173,173,173,173,198,230,257,257,25,25]

clc; clear;
addpath(fileparts(mfilename('fullpath')));  %#ok<*MCAP>

% 热力学参数
xm = [6.6857e-04, 2.1313e+04, ...
      8.1830e-04, 1.2193e+03, ...
      9.8977e-04, 7.2606e+02, ...
      8.6450e-04, 6.0728e+02, ...
      5.4317e-04, 9.9975e+02];

F1 = [25, 173, 198, 230, 257, 25];
v = 78/60;  % cm/s

[T_center, T_oven, t_full] = solve_heat(F1, v, xm);

% 截取传感器工作后的数据
T_s = T_center(T_center >= 30);
idx0 = find(T_center >= 30, 1);

% 指定位置
FL = 25; L = 30.5; G = 5;
x = [FL + 2*(L+G) + L/2;               % 小温区3中点
     FL + 5*L + 5*G + L/2;              % 小温区6中点
     FL + 5*L + 5*G + L + G + L/2;      % 小温区7中点
     FL + 5*L + 5*G + L + G + L + G + L]; % 小温区8结束
t_pos = x / v;

fprintf('问题1: 给定参数的炉温曲线\n');
fprintf('v=78 cm/min, T=[173,173,173,173,173,198,230,257,257,25,25]\n\n');
fprintf('指定位置温度:\n');
names = {'小温区3中点', '小温区6中点', '小温区7中点', '小温区8结束处'};
for i = 1:4
    T_val = interp1(t_full, T_center, t_pos(i), 'linear');
    fprintf('  %s: x=%.1fcm, t=%.1fs, T=%.1f°C\n', names{i}, x(i), t_pos(i), T_val);
end

m = analyze_curve(T_s);
fprintf('\n炉温曲线指标:\n');
fprintf('  峰值温度: %.1f°C\n', m.Tmax);
fprintf('  最大斜率: %.3f°C/s\n', m.max_slope);
fprintf('  150-190°C时间: %.1fs\n', m.t_150_190);
fprintf('  >217°C时间: %.1fs\n', m.t_above);
fprintf('  >217°C到峰值面积: %.1f°C·s\n', m.area);
fprintf('  制程界限: %s\n', cond(check_constraints(m), '满足', '不满足'));

% 输出 result.csv
rootDir = fullfile(fileparts(mfilename('fullpath')), '..');
t_out = (0:length(T_s)-1)' * 0.5;
T_out = T_s;
fid = fopen(fullfile(rootDir, 'result', 'figures', 'result.csv'), 'w');
fprintf(fid, '时间(s),温度(°C)\n');
for i = 1:length(t_out)
    fprintf(fid, '%.1f,%.4f\n', t_out(i), T_out(i));
end
fclose(fid);
fprintf('\nresult.csv 已保存 (%d行)\n', length(t_out));

% 绘图
figure('Position', [100, 100, 1200, 450]);
subplot(1,2,1); hold on;
h1 = plot(t_full, T_center, 'r-', 'LineWidth', 1.5);
h2 = plot(t_full, T_oven, 'b--', 'LineWidth', 1);
h3 = yline(217, 'g--', 'LineWidth', 1);
h4 = yline(30, ':k', 'LineWidth', 1);
for i = 1:4
    T_val = interp1(t_full, T_center, t_pos(i), 'linear');
    plot(t_pos(i), T_val, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k');
end
hold off; xlabel('时间 (s)'); ylabel('温度 (°C)');
title('问题1: 炉温曲线 (时间-温度)');
legend([h1, h2, h3, h4], {'焊接中心温度', '炉内环境温度', 'T=217°C', 'T=30°C'}, ...
    'Location', 'southeast', 'FontSize', 9);
grid on;

subplot(1,2,2); hold on;
h1 = plot(v*t_full, T_center, 'r-', 'LineWidth', 1.5);
h2 = plot(v*t_full, T_oven, 'b--', 'LineWidth', 1);
h3 = yline(217, 'g--', 'LineWidth', 1);
for i = 1:4
    T_val = interp1(t_full, T_center, t_pos(i), 'linear');
    plot(x(i), T_val, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k');
end
hold off; xlabel('位置 (cm)'); ylabel('温度 (°C)');
title('问题1: 炉温曲线 (位置-温度)');
legend([h1, h2, h3], {'焊接中心温度', '炉内环境温度', 'T=217°C'}, ...
    'Location', 'southeast', 'FontSize', 9);
grid on;

print(gcf, fullfile(fileparts(mfilename('fullpath')), '..', 'result', 'figures', 'q1_furnace_curve.png'), '-dpng', '-r200');

function s = cond(c, t, f)
    if c, s = t; else, s = f; end
end
