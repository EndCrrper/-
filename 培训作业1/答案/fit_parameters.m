%% ============================================================
% 参数估计: 利用实验数据拟合热力学参数
% ============================================================
% 实验条件: v=70 cm/min, T=[175,175,175,175,175,195,235,255,255,25,25]
% 待估参数: [a1,h1, a2,h2, a3,h3, a4,h4, a5,h5] (10维)
% 方法: 带自然选择的粒子群优化 (SelPSO/CLSPSO)
% ============================================================

clc; clear; close all;

%% ---- 加载实验数据 ----
filename = '..\题目\附件.xlsx';
if ~exist(filename, 'file')
    filename = '附件.xlsx';
end

fprintf('========== 参数估计: PSO 拟合热力学参数 ==========\n');

try
    E = readmatrix(filename);
    t_exp = E(:, 1);     % 实验时间 (s)
    T_exp = E(:, 2);     % 实验温度 (°C)
    fprintf('成功加载实验数据: %d 个数据点\n', length(T_exp));
catch
    error('无法加载附件.xlsx，请确认文件路径。');
end

%% ---- 实验条件 ----
F_exp = [25, 175, 195, 235, 255, 25];  % 实验温度设定
v_exp = 70 / 60;                         % 实验速度 (cm/s)
T0 = 19;  % 传感器启动时间 (s)，实验数据显示约在 t=19s 时 T>=30°C

fprintf('实验条件: v=70 cm/min, T=[175,175,175,175,175,195,235,255,255,25,25]\n');

%% ---- PSO 参数设定 ----
% 变量范围 [a1, h1, a2, h2, a3, h3, a4, h4, a5, h5]
lb = [4e-4, 20000, 4e-4, 700, 4e-4, 700, 4e-4, 200, 4e-4, 700];
ub = [9e-4, 30000, 2e-3, 1500, 2e-3, 1500, 2e-3, 1000, 2e-3, 1500];

N = 60;          % 粒子数
w = 0.8;         % 惯性权重
c1 = 2.05;       % 个体学习因子
c2 = 2.05;       % 社会学习因子
M = 1000;        % 迭代次数
D = 10;          % 变量维度

fprintf('PSO参数: 粒子数=%d, 迭代次数=%d, 变量维度=%d\n', N, M, D);

%% ---- SelPSO 优化 ----
tic;
[best_xm, best_fval] = sel_pso_fit(@objective_fit, N, w, c1, c2, ub, lb, M, D, ...
                                     F_exp, v_exp, t_exp, T_exp, T0);
elapsed = toc;

fprintf('\n优化完成! 耗时: %.2f 秒\n', elapsed);
fprintf('最优目标函数值(SSE): %.4f\n', best_fval);

%% ---- 输出结果 ----
fprintf('\n========== 拟合结果 (热力学参数) ==========\n');
fprintf('  区域          |  a (cm/s^0.5)  |  h (1/cm)\n');
fprintf('  --------------|----------------|----------------\n');
zone_names = {'预热区(炉前~温区5)', '恒温区(温区6)', ...
              '回流升温(温区7)', '回流峰值(温区8~9)', '冷却区(温区10~11)'};
for i = 1:5
    fprintf('  %-14s |  %.6e  |  %.4e\n', ...
        zone_names{i}, best_xm(2*i-1), best_xm(2*i));
end

% 计算 a² 值
fprintf('\n  区域          |  a² = k/(ρc) (cm²/s)\n');
fprintf('  --------------|----------------\n');
for i = 1:5
    fprintf('  %-14s |  %.4e\n', zone_names{i}, best_xm(2*i-1)^2);
end

%% ---- 验证拟合效果 ----
[T_sim, ~, ~] = solve_heat(F_exp, v_exp, best_xm);
idx_start = find(T_sim >= 30, 1, 'first');
if isempty(idx_start), idx_start = 1; end

% 插值到实验时间点
T_sim_interp = interp1((0:length(T_sim)-1)*0.5, T_sim, t_exp, 'linear');

% 误差分析
errors = T_sim_interp - T_exp;
SSE = sum(errors.^2);
RMSE = sqrt(mean(errors.^2));
MAE = mean(abs(errors));
R2 = 1 - sum(errors.^2) / sum((T_exp - mean(T_exp)).^2);

fprintf('\n--- 拟合精度 ---\n');
fprintf('  SSE:  %.4f\n', SSE);
fprintf('  RMSE: %.4f °C\n', RMSE);
fprintf('  MAE:  %.4f °C\n', MAE);
fprintf('  R²:   %.4f\n', R2);
fprintf('  最大误差: %.4f °C\n', max(abs(errors)));

%% ---- 绘图 ----
figure('Position', [100, 100, 1200, 500]);

% 子图1: 模拟 vs 实验温度
subplot(1,2,1);
plot(t_exp, T_exp, 'b.', 'MarkerSize', 8);
hold on;
plot(t_exp, T_sim_interp, 'r-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('温度 (°C)');
title(sprintf('实验温度 vs 模拟温度 (R²=%.4f, RMSE=%.2f°C)', R2, RMSE));
legend('实验数据', '模拟值', 'Location', 'southeast');
grid on;

% 子图2: 残差分析
subplot(1,2,2);
plot(t_exp, errors, 'k.', 'MarkerSize', 8);
hold on;
yline(0, 'r-', 'LineWidth', 0.8);
yline(3*RMSE, 'b--', 'LineWidth', 0.8);
yline(-3*RMSE, 'b--', 'LineWidth', 0.8);
xlabel('时间 (s)'); ylabel('残差 (°C)');
title(sprintf('残差分析 (MAE=%.2f°C, Max=%.2f°C)', MAE, max(abs(errors))));
legend('残差', '零线', '±3RMSE', 'Location', 'southeast');
grid on;

saveas(gcf, 'parameter_fit.png');
fprintf('\n图已保存: parameter_fit.png\n');

% 保存参数到文件
save('fitted_parameters.mat', 'best_xm', 'best_fval', 'RMSE', 'R2');
fprintf('参数已保存到 fitted_parameters.mat\n');
fprintf('\n参数估计完成!\n');

%% ==================== 目标函数 ====================
function SSE = objective_fit(xm, F_exp, v_exp, t_exp, T_exp, T0)
    % 计算模拟温度与实验温度的误差平方和

    [T_sim, ~, ~] = solve_heat(F_exp, v_exp, xm);
    idx_start = find(T_sim >= 30, 1, 'first');
    if isempty(idx_start), idx_start = 1; end

    % 截取传感器启动后的数据
    T_sim_sensor = T_sim(idx_start:end);
    t_sim = (0:length(T_sim_sensor)-1) * 0.5;

    % 插值到实验时间点
    T_sim_interp = interp1(t_sim, T_sim_sensor, t_exp, 'linear', NaN);

    if any(isnan(T_sim_interp))
        SSE = Inf; return;
    end

    errors = T_sim_interp - T_exp;
    SSE = sum(errors.^2);
end

%% ==================== SelPSO 算法 (10维版本) ====================
function [xm_best, fv_best] = sel_pso_fit(fitness, N, w, c1, c2, xmax, xmin, M, D, ...
                                          F_exp, v_exp, t_exp, T_exp, T0)
    % 带自然选择的粒子群优化

    Vmax = 0.2 * (xmax - xmin);

    % 初始化
    x = zeros(N, D); v = zeros(N, D);
    for i = 1:N
        x(i, :) = xmin + rand(1, D) .* (xmax - xmin);
        v(i, :) = Vmax .* (-1 + 2 * rand(1, D));
    end

    p = zeros(N, 1); y = zeros(N, D);
    for i = 1:N
        p(i) = fitness(x(i, :), F_exp, v_exp, t_exp, T_exp, T0);
        y(i, :) = x(i, :);
    end

    pg = p(N); px = x(N, :);
    for i = 1:N-1
        if p(i) < pg
            pg = p(i); px = x(i, :);
        end
    end

    f = zeros(N, 1);
    for t = 1:M
        for i = 1:N
            v(i, :) = w * v(i, :) + c1 * rand(1, D) .* (y(i, :) - x(i, :)) ...
                                   + c2 * rand(1, D) .* (px - x(i, :));
            for j = 1:D
                if v(i, j) > Vmax(j), v(i, j) = Vmax(j); end
                if v(i, j) < -Vmax(j), v(i, j) = -Vmax(j); end
            end
            x(i, :) = x(i, :) + v(i, :);
            if all(x(i, :) <= xmax) && all(x(i, :) >= xmin)
                f(i) = fitness(x(i, :), F_exp, v_exp, t_exp, T_exp, T0);
            else
                f(i) = Inf;
            end
            if f(i) < p(i)
                p(i) = f(i); y(i, :) = x(i, :);
            end
            if p(i) < pg
                pg = p(i); px = y(i, :);
            end
        end

        % 自然选择
        [~, sort_idx] = sort(f);
        ex_index = round((N - 1) / 2);
        x(sort_idx(N - ex_index + 1:N), :) = x(sort_idx(1:ex_index), :);
        v(sort_idx(N - ex_index + 1:N), :) = v(sort_idx(1:ex_index), :);

        if mod(t, 200) == 0
            fprintf('  迭代 %3d/%d, SSE: %.4f\n', t, M, pg);
        end
    end

    xm_best = px';
    fv_best = pg;
end
