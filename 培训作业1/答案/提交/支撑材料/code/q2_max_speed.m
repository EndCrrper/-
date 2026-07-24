%% 问题2: 最大传送带速度
% T=[182,182,182,182,182,203,237,254,254,25,25]
% 二分搜索

clc; clear;
addpath(fileparts(mfilename('fullpath')));

xm = [6.6857e-04, 2.1313e+04, ...
      8.1830e-04, 1.2193e+03, ...
      9.8977e-04, 7.2606e+02, ...
      8.6450e-04, 6.0728e+02, ...
      5.4317e-04, 9.9975e+02];

F2 = [25, 182, 203, 237, 254, 25];

fprintf('问题2: 最大传送带速度\n');
fprintf('T=[182,182,182,182,182,203,237,254,254,25,25]\n\n');

% 速度扫描
fprintf('速度(cm/min) | 峰值(°C) | 斜率(°C/s) | 150-190(s) | >217(s) | 可行\n');
fprintf('-------------|----------|------------|------------|---------|-----\n');
for v_cm = 65:5:100
    [T_c, ~, ~] = solve_heat(F2, v_cm/60, xm);
    T_s = T_c(T_c >= 30);
    m = analyze_curve(T_s);
    ok = check_constraints(m);
    fprintf('  %6d     |  %6.1f  |  %7.3f   |  %7.1f   |  %5.1f  |  %d\n', ...
        v_cm, m.Tmax, m.max_slope, m.t_150_190, m.t_above, ok);
end

% 二分搜索
v_lo = 65; v_hi = 100;
[T_c, ~, ~] = solve_heat(F2, v_hi/60, xm);
T_s = T_c(T_c >= 30);
if ~isempty(T_s) && check_constraints(analyze_curve(T_s))
    v_max = v_hi;
else
    while v_hi - v_lo > 1e-4
        v_mid = (v_lo + v_hi) / 2;
        [T_c, ~, ~] = solve_heat(F2, v_mid/60, xm);
        T_s = T_c(T_c >= 30);
        if ~isempty(T_s) && check_constraints(analyze_curve(T_s))
            v_lo = v_mid;
        else
            v_hi = v_mid;
        end
    end
    v_max = v_lo;
end

fprintf('\n最大允许速度: %.4f cm/min\n', v_max);

% 验证
[T_c, ~, ~] = solve_heat(F2, v_max/60, xm);
T_s = T_c(T_c >= 30);
m = analyze_curve(T_s);
fprintf('  Tmax=%.1f°C, 斜率=%.3f°C/s, 150-190=%.1fs, >217=%.1fs\n', ...
    m.Tmax, m.max_slope, m.t_150_190, m.t_above);

% 四象限图
figure('Position', [100, 100, 1000, 800]);
v_list = 65:5:100;
for i = 1:length(v_list)
    [T_c, ~, ~] = solve_heat(F2, v_list(i)/60, xm);
    T_s = T_c(T_c >= 30);
    s(i) = analyze_curve(T_s);
end

subplot(2,2,1); hold on;
plot(v_list, [s.max_slope], 'b-o', 'MarkerSize', 6, 'LineWidth', 1.2);
yline(3, 'r--', 'LineWidth', 1.2);
xlabel('速度 (cm/min)'); ylabel('斜率 (°C/s)');
title('(a) 最大速率 vs 速度'); legend('模拟值', '上限 3°C/s', 'FontSize', 8); grid on;

subplot(2,2,2); hold on;
plot(v_list, [s.Tmax], 'r-o', 'MarkerSize', 6, 'LineWidth', 1.2);
yline(250, 'r--', 'LineWidth', 1.2);
yline(240, '--', 'Color', [1 0.5 0], 'LineWidth', 1.2);
xlabel('速度 (cm/min)'); ylabel('峰值温度 (°C)');
title('(b) 峰值温度 vs 速度'); legend('模拟值', '上限 250', '下限 240', 'FontSize', 8); grid on;

subplot(2,2,3); hold on;
plot(v_list, [s.t_above], 'g-o', 'MarkerSize', 6, 'LineWidth', 1.2);
yline(90, 'r--', 'LineWidth', 1.2);
yline(40, '--', 'Color', [1 0.5 0], 'LineWidth', 1.2);
xlabel('速度 (cm/min)'); ylabel('时间 (s)');
title('(c) >217°C 时间 vs 速度'); legend('模拟值', '上限 90s', '下限 40s', 'FontSize', 8); grid on;

subplot(2,2,4); hold on;
plot(v_list, [s.t_150_190], 'm-o', 'MarkerSize', 6, 'LineWidth', 1.2);
yline(120, 'r--', 'LineWidth', 1.2);
yline(60, '--', 'Color', [1 0.5 0], 'LineWidth', 1.2);
xlabel('速度 (cm/min)'); ylabel('时间 (s)');
title('(d) 150-190°C 时间 vs 速度'); legend('模拟值', '上限 120s', '下限 60s', 'FontSize', 8); grid on;

sgtitle(sprintf('问题2: 速度扫描分析  ->  v_{max} = %.1f cm/min', v_max), 'FontSize', 13);
print(gcf, fullfile(fileparts(mfilename('fullpath')), '..', 'result', 'figures', 'q2_speed_analysis.png'), '-dpng', '-r300');
