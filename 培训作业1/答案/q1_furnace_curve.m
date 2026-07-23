%% 问题1: 给定参数的炉温曲线
% v=78 cm/min, T=[173,173,173,173,173,198,230,257,257,25,25]

clc; clear;

% 热力学参数
xm = [6.677003439269690e-04, 2.999997701878186e+04, ...
      8.120630727730775e-04, 1.499997442761376e+03, ...
      9.300295943899003e-04, 1.389867556248244e+03, ...
      8.343291166559698e-04, 6.898624109710258e+02, ...
      5.307244068332524e-04, 1.265962465478697e+03];

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
t_out = (0:length(T_s)-1)' * 0.5;
T_out = T_s;
fid = fopen('result.csv', 'w');
fprintf(fid, '时间(s),温度(°C)\n');
for i = 1:length(t_out)
    fprintf(fid, '%.1f,%.4f\n', t_out(i), T_out(i));
end
fclose(fid);
fprintf('\nresult.csv 已保存 (%d行)\n', length(t_out));

% 绘图
figure;
subplot(1,2,1);
plot(t_full, T_center, 'r-', 'LineWidth', 1.5); hold on;
plot(t_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--'); yline(30, ':k');
for i = 1:4
    T_val = interp1(t_full, T_center, t_pos(i), 'linear');
    plot(t_pos(i), T_val, 'o', 'MarkerSize', 8);
end
hold off; xlabel('时间(s)'); ylabel('温度(°C)');
title('问题1: 炉温曲线');
legend('焊接中心温度', '炉内环境温度', 'T=217°C', 'T=30°C', '关键位置');
grid on;

subplot(1,2,2);
plot(v*t_full, T_center, 'r-', 'LineWidth', 1.5); hold on;
plot(v*t_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--');
for i = 1:4
    T_val = interp1(t_full, T_center, t_pos(i), 'linear');
    plot(x(i), T_val, 'o', 'MarkerSize', 8);
end
hold off; xlabel('位置(cm)'); ylabel('温度(°C)');
title('位置-温度曲线'); legend('焊接中心温度', '炉内环境温度', 'T=217°C', '关键位置');
grid on;

saveas(gcf, 'q1_furnace_curve.png');

function s = cond(c, t, f)
    if c, s = t; else, s = f; end
end
