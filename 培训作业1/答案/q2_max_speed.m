%% ============================================================
% 问题2: 确定允许的最大传送带过炉速度
% ============================================================
% 条件: T=[182,182,182,182,182,203,237,254,254,25,25]
% 目标: 在满足制程界限的前提下，最大化传送带速度
% 方法: 二分搜索 (利用速度-温度指标的单调性)
% ============================================================

clc; clear; close all;

%% ---- 参数设定 ----
% 热力学参数
xm = [6.677003439269690e-04, 2.999997701878186e+04, ...
      8.120630727730775e-04, 1.499997442761376e+03, ...
      9.300295943899003e-04, 1.389867556248244e+03, ...
      8.343291166559698e-04, 6.898624109710258e+02, ...
      5.307244068332524e-04, 1.265962465478697e+03];

F2 = [25, 182, 203, 237, 254, 25];  % 问题2温度设定
v_range = [65, 100];                 % 速度搜索范围 (cm/min)

fprintf('========== 问题2: 最大传送带速度 ==========\n');
fprintf('温度设定: T1-5=%d, T6=%d, T7=%d, T8-9=%d°C\n', ...
    F2(2), F2(3), F2(4), F2(5));

%% ---- 速度扫描 (粗步长5 cm/min) ----
fprintf('\n--- 速度扫描 (65:5:100 cm/min) ---\n');
fprintf('  速度(cm/min) | 峰值(°C) | 斜率(°C/s) | 150-190(s) | >217(s) | 可行\n');
fprintf('  -------------|-----------|------------|------------|----------|-----\n');

v_scan = 65:5:100;
scan_Tmax = zeros(size(v_scan));
scan_slope = zeros(size(v_scan));
scan_t1 = zeros(size(v_scan));
scan_t2 = zeros(size(v_scan));
scan_ok = false(size(v_scan));

for i = 1:length(v_scan)
    v_try = v_scan(i) / 60;
    [T_c, ~, ~] = solve_heat(F2, v_try, xm);
    T_s = T_c(T_c >= 30);
    m = analyze_curve(T_s);
    ok = check_constraints(m);

    scan_Tmax(i) = m.Tmax;
    scan_slope(i) = m.max_slope;
    scan_t1(i) = m.t_150_190;
    scan_t2(i) = m.t_above;
    scan_ok(i) = ok;

    fprintf('  %8d     | %6.1f   | %7.3f    | %7.1f    | %6.1f  |  %s\n', ...
        v_scan(i), m.Tmax, m.max_slope, m.t_150_190, m.t_above, ...
        ternary(ok, '✓', '✗'));
end

%% ---- 二分搜索最大速度 ----
fprintf('\n--- 二分搜索最大速度 ---\n');

v_lo = 65; v_hi = 100;

% 先检查上界是否可行
[T_c, ~, ~] = solve_heat(F2, v_hi/60, xm);
T_s = T_c(T_c >= 30);
if ~isempty(T_s) && check_constraints(analyze_curve(T_s))
    v_max = v_hi;
    fprintf('上界 %.0f cm/min 可行，即为最大速度\n', v_hi);
else
    iter = 0;
    while (v_hi - v_lo) > 1e-4
        iter = iter + 1;
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
    fprintf('二分搜索完成 (%d 次迭代)\n', iter);
end

fprintf('\n>>> 允许的最大传送带过炉速度: %.3f cm/min <<<\n', v_max);

%% ---- 验证最优解 ----
[T_c, ~, ~] = solve_heat(F2, v_max/60, xm);
T_s = T_c(T_c >= 30);
m = analyze_curve(T_s);
ok = check_constraints(m);

fprintf('\n--- 最优解验证 (v = %.3f cm/min) ---\n', v_max);
fprintf('  峰值温度:        %.1f °C  [要求: 240~250]  %s\n', m.Tmax, check_mark(m.Tmax, 240, 250));
fprintf('  最大斜率:        %.3f °C/s [要求: ≤3]      %s\n', m.max_slope, check_mark(m.max_slope, -inf, 3));
fprintf('  150-190°C时间:   %.1f s   [要求: 60~120]  %s\n', m.t_150_190, check_mark(m.t_150_190, 60, 120));
fprintf('  >217°C时间:      %.1f s   [要求: 40~90]   %s\n', m.t_above, check_mark(m.t_above, 40, 90));

fprintf('\n所有约束满足: %s\n', ternary(ok, '是 ✓', '否 ✗'));

%% ---- 绘图: 速度扫描四象限图 ----
figure('Position', [100, 100, 1200, 900]);

subplot(2,2,1);
plot(v_scan, scan_slope, 'b-o', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on; yline(3, 'r--', 'LineWidth', 1.2);
xlabel('速度 (cm/min)'); ylabel('最大斜率 (°C/s)');
title('(a) 最大速率 vs 速度'); legend('模拟值', '上限 ≤3°C/s'); grid on;

subplot(2,2,2);
plot(v_scan, scan_Tmax, 'r-o', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on;
yline(250, 'r--'); yline(240, '--', 'Color', [1 0.5 0]);
xlabel('速度 (cm/min)'); ylabel('峰值温度 (°C)');
title('(b) 峰值温度 vs 速度'); legend('模拟值', '上限250', '下限240'); grid on;

subplot(2,2,3);
plot(v_scan, scan_t2, 'g-o', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on;
yline(90, 'r--'); yline(40, '--', 'Color', [1 0.5 0]);
xlabel('速度 (cm/min)'); ylabel('时间 (s)');
title('(c) >217°C时间 vs 速度'); legend('模拟值', '上限90s', '下限40s'); grid on;

subplot(2,2,4);
plot(v_scan, scan_t1, 'm-o', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on;
yline(120, 'r--'); yline(60, '--', 'Color', [1 0.5 0]);
xlabel('速度 (cm/min)'); ylabel('时间 (s)');
title('(d) 150-190°C时间 vs 速度'); legend('模拟值', '上限120s', '下限60s'); grid on;

sgtitle(sprintf('问题2: 速度扫描分析 → 最大速度 v_{max} = %.1f cm/min', v_max));
saveas(gcf, 'q2_speed_analysis.png');
fprintf('\n图已保存: q2_speed_analysis.png\n');
fprintf('问题2 求解完成!\n');

%% ---- 辅助函数 ----
function s = ternary(cond, t, f)
    if cond, s = t; else, s = f; end
end

function m = check_mark(val, lo, hi)
    if val >= lo && val <= hi
        m = '✓';
    else
        m = '✗';
    end
end
