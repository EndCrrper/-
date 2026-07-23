%% 问题2: 最大传送带速度
% T=[182,182,182,182,182,203,237,254,254,25,25]
% 二分搜索

clc; clear;

xm = [6.677003439269690e-04, 2.999997701878186e+04, ...
      8.120630727730775e-04, 1.499997442761376e+03, ...
      9.300295943899003e-04, 1.389867556248244e+03, ...
      8.343291166559698e-04, 6.898624109710258e+02, ...
      5.307244068332524e-04, 1.265962465478697e+03];

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

fprintf('\n最大允许速度: %.2f cm/min\n', v_max);

% 验证
[T_c, ~, ~] = solve_heat(F2, v_max/60, xm);
T_s = T_c(T_c >= 30);
m = analyze_curve(T_s);
fprintf('  Tmax=%.1f°C, 斜率=%.3f°C/s, 150-190=%.1fs, >217=%.1fs\n', ...
    m.Tmax, m.max_slope, m.t_150_190, m.t_above);

% 四象限图
figure;
v_list = 65:5:100;
for i = 1:length(v_list)
    [T_c, ~, ~] = solve_heat(F2, v_list(i)/60, xm);
    T_s = T_c(T_c >= 30);
    s(i) = analyze_curve(T_s);
end

subplot(2,2,1);
plot(v_list, [s.max_slope], 'b-o', 'MarkerSize', 5); hold on;
yline(3, 'r--'); xlabel('速度(cm/min)'); ylabel('斜率(°C/s)');
title('最大速率 vs 速度'); grid on;

subplot(2,2,2);
plot(v_list, [s.Tmax], 'r-o', 'MarkerSize', 5); hold on;
yline(250, 'r--'); yline(240, '--', 'Color', [1 0.5 0]);
xlabel('速度(cm/min)'); ylabel('温度(°C)');
title('峰值温度 vs 速度'); grid on;

subplot(2,2,3);
plot(v_list, [s.t_above], 'g-o', 'MarkerSize', 5); hold on;
yline(90, 'r--'); yline(40, '--', 'Color', [1 0.5 0]);
xlabel('速度(cm/min)'); ylabel('时间(s)');
title('>217°C时间 vs 速度'); grid on;

subplot(2,2,4);
plot(v_list, [s.t_150_190], 'm-o', 'MarkerSize', 5); hold on;
yline(120, 'r--'); yline(60, '--', 'Color', [1 0.5 0]);
xlabel('速度(cm/min)'); ylabel('时间(s)');
title('150-190°C时间 vs 速度'); grid on;

sgtitle(sprintf('问题2: v_{max}=%.1f cm/min', v_max));
saveas(gcf, 'q2_speed_analysis.png');
