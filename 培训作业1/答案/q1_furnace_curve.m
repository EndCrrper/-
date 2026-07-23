%% ============================================================
% 问题1: 给定参数的炉温曲线计算
% ============================================================
% 条件: v=78 cm/min, T=[173,173,173,173,173,198,230,257,257,25,25]
% 输出: 炉温曲线图、指定位置温度、result.csv
% ============================================================

clc; clear; close all;

%% ---- 参数设定 ----
% 热力学参数 (PSO 拟合结果, 共3组可选)
xm_all = [
    % 组1: 主参数组
    6.677003439269690e-04, 2.999997701878186e+04, ...
    8.120630727730775e-04, 1.499997442761376e+03, ...
    9.300295943899003e-04, 1.389867556248244e+03, ...
    8.343291166559698e-04, 6.898624109710258e+02, ...
    5.307244068332524e-04, 1.265962465478697e+03;
    % 组2: 备选参数组1
    6.680738508034592e-04, 2.665015993703618e+04, ...
    8.162196029159215e-04, 1.199766752638612e+03, ...
    9.827794831794097e-04, 8.065131954728832e+02, ...
    8.362762276929443e-04, 7.385816657683414e+02, ...
    5.271121937960067e-04, 1.363617464878294e+03;
    % 组3: 备选参数组2
    6.683140781153043e-04, 2.498727636628837e+04, ...
    8.076275611754757e-04, 1.431525270900793e+03, ...
    9.743913118484225e-04, 8.279199794106737e+02, ...
    8.492734595661612e-04, 6.547702252291799e+02, ...
    5.286841991732021e-04, 1.337603876051590e+03
];
xm = xm_all(1, :);  % 选用第1组参数

% 问题1给定条件
F1 = [25, 173, 198, 230, 257, 25];  % 各段设定温度
v_cm_min = 78;                        % 传送带速度 (cm/min)
v_cms = v_cm_min / 60;                % 转换为 cm/s

fprintf('========== 问题1: 给定参数的炉温曲线 ==========\n');
fprintf('速度: %d cm/min = %.2f cm/s\n', v_cm_min, v_cms);
fprintf('温度设定: T1-5=%d, T6=%d, T7=%d, T8-9=%d, T10-11=%d°C\n', ...
    F1(2), F1(3), F1(4), F1(5), F1(6));

%% ---- 求解 ----
[T_center, T_oven, t_full] = solve_heat(F1, v_cms, xm);

% 截取 T>=30°C 的数据 (从传感器启动开始)
T_sensor = T_center(T_center >= 30);
idx_start = find(T_center >= 30, 1, 'first');

% 时间向量
t_sensor = (0:length(T_sensor)-1) * 0.5;

%% ---- 关键位置温度 ----
% 各位置距离计算
FL = 25; ZL = 30.5; GL = 5;
pos_names = {'小温区3中点', '小温区6中点', '小温区7中点', '小温区8结束处'};
pos_x = zeros(4, 1);
pos_x(1) = FL + 2*(ZL+GL) + ZL/2;       % 小温区3中点
pos_x(2) = FL + 5*ZL + 5*GL + ZL/2;     % 小温区6中点
pos_x(3) = FL + 5*ZL + 5*GL + ZL + GL + ZL/2;  % 小温区7中点
pos_x(4) = FL + 5*ZL + 5*GL + ZL + GL + ZL + GL + ZL;  % 小温区8结束

pos_t = pos_x / v_cms;  % 到达各位置的时间

fprintf('\n--- 指定位置温度 ---\n');
for i = 1:4
    T_val = interp1(t_full, T_center, pos_t(i), 'linear');
    fprintf('  %s: x=%.1f cm, t=%.1f s, T=%.1f °C\n', ...
        pos_names{i}, pos_x(i), pos_t(i), T_val);
end

%% ---- 炉温曲线指标 ----
m = analyze_curve(T_sensor);
fprintf('\n--- 炉温曲线指标 ---\n');
fprintf('  峰值温度:          %.1f °C\n', m.Tmax);
fprintf('  最大升/降温速率:   %.3f °C/s\n', m.max_slope);
fprintf('  150-190°C(升温):   %.1f s\n', m.t_150_190);
fprintf('  >217°C时间:        %.1f s\n', m.t_above);
fprintf('  >217°C到峰值面积:  %.1f °C·s\n', m.area);

ok = check_constraints(m);
fprintf('  制程界限:          %s\n', ternary(ok, '满足', '不满足'));

%% ---- 输出 result.csv ----
fprintf('\n生成 result.csv ...\n');
T_out = T_center(idx_start:end);
t_out = t_full(idx_start:end);
data_table = table(t_out(:), T_out(:), 'VariableNames', {'时间_s', '温度_C'});
writetable(data_table, 'result.csv');
fprintf('  已保存 %d 行数据到 result.csv\n', length(T_out));

%% ---- 绘图 ----
figure('Position', [100, 100, 1400, 500]);

% 子图1: 时间-温度曲线
subplot(1,2,1);
hold on;
plot(t_sensor, T_sensor, 'r-', 'LineWidth', 1.5);
plot(t_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--', 'LineWidth', 0.8);
yline(30, ':k', 'LineWidth', 0.5);
% 标注关键位置
for i = 1:4
    T_val = interp1(t_full, T_center, pos_t(i), 'linear');
    plot(pos_t(i), T_val, 'o', 'MarkerSize', 8, 'LineWidth', 1.5);
end
hold off;
xlabel('时间 (s)'); ylabel('温度 (°C)');
title('问题1: 炉温曲线 (时间-温度)');
legend('焊接区域中心温度', '炉内环境温度', 'T=217°C', 'T=30°C', '关键位置', ...
    'Location', 'southeast');
grid on;

% 子图2: 位置-温度曲线
subplot(1,2,2);
x_full = v_cms * t_full;
x_sensor = v_cms * t_sensor;
hold on;
plot(x_sensor, T_sensor, 'r-', 'LineWidth', 1.5);
plot(x_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--', 'LineWidth', 0.8);
for i = 1:4
    T_val = interp1(t_full, T_center, pos_t(i), 'linear');
    plot(pos_x(i), T_val, 'o', 'MarkerSize', 8, 'LineWidth', 1.5);
end
hold off;
xlabel('位置 (cm)'); ylabel('温度 (°C)');
title('问题1: 炉温曲线 (位置-温度)');
legend('焊接区域中心温度', '炉内环境温度', 'T=217°C', '关键位置', ...
    'Location', 'southeast');
grid on;

saveas(gcf, 'q1_furnace_curve.png');
fprintf('  图已保存: q1_furnace_curve.png\n');

fprintf('\n问题1 求解完成!\n');

%% ---- 辅助函数 ----
function s = ternary(cond, t, f)
    if cond, s = t; else, s = f; end
end
