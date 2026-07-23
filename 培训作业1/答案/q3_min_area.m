%% ============================================================
% 问题3: 最小化超过217°C到峰值温度的面积
% ============================================================
% 优化变量: [T1_5, T6, T7, T8_9, v(cm/s)]
% 目标函数: min S = 面积(超过217°C到峰值)
% 约束条件: 满足4条制程界限 + 变量范围
% 方法: 带自然选择的粒子群优化 (SelPSO)
% ============================================================

clc; clear; close all;

%% ---- 参数设定 ----
% 热力学参数
xm = [6.677003439269690e-04, 2.999997701878186e+04, ...
      8.120630727730775e-04, 1.499997442761376e+03, ...
      9.300295943899003e-04, 1.389867556248244e+03, ...
      8.343291166559698e-04, 6.898624109710258e+02, ...
      5.307244068332524e-04, 1.265962465478697e+03];

% 变量范围 [T1_5, T6, T7, T8_9, v(cm/s)]
lb = [165, 185, 225, 245, 65/60];     % 下界
ub = [185, 205, 245, 265, 100/60];    % 上界

% PSO 参数
N = 60;          % 粒子数
w = 0.8;         % 惯性权重
c1 = 2.0;        % 个体学习因子
c2 = 2.0;        % 社会学习因子
M = 500;         % 最大迭代次数
D = 5;           % 决策变量维度

fprintf('========== 问题3: 最小化 >217°C 到峰值的面积 ==========\n');
fprintf('优化方法: 带自然选择的粒子群优化 (SelPSO)\n');
fprintf('粒子数: %d, 迭代次数: %d\n', N, M);
fprintf('变量范围: T1-5∈[%d,%d], T6∈[%d,%d], T7∈[%d,%d], T8-9∈[%d,%d], v∈[%d,%d] cm/min\n', ...
    lb(1), ub(1), lb(2), ub(2), lb(3), ub(3), lb(4), ub(4), round(lb(5)*60), round(ub(5)*60));

%% ---- SelPSO 优化 ----
tic;
[best_x, best_fval] = sel_pso(@objective_area, N, w, c1, c2, ub, lb, M, D, xm);
elapsed = toc;

fprintf('\n优化完成! 耗时: %.2f 秒\n', elapsed);

%% ---- 输出最优解 ----
fprintf('\n========== 最优解 ==========\n');
fprintf('  小温区1~5温度:  %.1f °C\n', best_x(1));
fprintf('  小温区6温度:    %.1f °C\n', best_x(2));
fprintf('  小温区7温度:    %.1f °C\n', best_x(3));
fprintf('  小温区8~9温度:  %.1f °C\n', best_x(4));
fprintf('  传送带速度:     %.2f cm/min\n', best_x(5)*60);
fprintf('  最小面积:       %.1f °C·s\n', best_fval);

%% ---- 最优解验证 ----
F_opt = [25, best_x(1), best_x(2), best_x(3), best_x(4), 25];
[T_c, ~, T_oven] = solve_heat(F_opt, best_x(5), xm);
T_s = T_c(T_c >= 30);
m = analyze_curve(T_s);

fprintf('\n--- 约束验证 ---\n');
fprintf('  峰值温度:        %.1f °C  [240~250]  %s\n', m.Tmax, ...
    check_mark(m.Tmax, 240, 250));
fprintf('  最大速率:        %.3f °C/s [≤3]       %s\n', m.max_slope, ...
    check_mark(m.max_slope, -inf, 3));
fprintf('  150-190°C时间:   %.1f s   [60~120]   %s\n', m.t_150_190, ...
    check_mark(m.t_150_190, 60, 120));
fprintf('  >217°C时间:      %.1f s   [40~90]    %s\n', m.t_above, ...
    check_mark(m.t_above, 40, 90));
fprintf('  所有约束满足: %s\n', ternary(check_constraints(m), '✓', '✗'));

%% ---- 绘图 ----
L_total = 25 + 11*30.5 + 10*5 + 25;
total_time = L_total / best_x(5);
dt = 0.5;
t_full = (0:length(T_c)-1) * dt;
x_full = best_x(5) * t_full;

figure('Position', [100, 100, 1400, 500]);

% 子图1: 时间-温度曲线 (标注面积)
subplot(1,2,1);
hold on;
% 填充超过217°C到峰值的面积
idx_217 = find(T_c >= 217, 1, 'first');
if isempty(idx_217), idx_217 = 1; end
[~, peak_idx] = max(T_c);
x_fill = t_full(idx_217:peak_idx);
y_top = T_c(idx_217:peak_idx);
y_bot = 217 * ones(size(x_fill));
fill([x_fill, fliplr(x_fill)], [y_top(:)', fliplr(y_bot(:)')], ...
    [1, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.6);

plot(t_full, T_c, 'r-', 'LineWidth', 1.5);
plot(t_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--', 'LineWidth', 0.8);
plot(t_full(peak_idx), T_c(peak_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
hold off;
xlabel('时间 (s)'); ylabel('温度 (°C)');
title(sprintf('问题3: 最优炉温曲线 (面积=%.1f °C·s)', best_fval));
legend('>217°C至峰值面积', '焊接区域中心温度', '炉内环境温度', 'T=217°C', '峰值');
grid on;

% 子图2: 位置-温度曲线
subplot(1,2,2);
plot(x_full, T_c, 'r-', 'LineWidth', 1.5);
hold on;
plot(x_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--', 'LineWidth', 0.8);
plot(x_full(peak_idx), T_c(peak_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
hold off;
xlabel('位置 (cm)'); ylabel('温度 (°C)');
title('问题3: 位置-温度曲线');
legend('焊接区域中心温度', '炉内环境温度', 'T=217°C', '峰值');
grid on;

saveas(gcf, 'q3_optimal.png');
fprintf('\n图已保存: q3_optimal.png\n');
fprintf('问题3 求解完成!\n');

%% ==================== 目标函数 ====================
function S = objective_area(x, xm)
    % 决策变量: [T1_5, T6, T7, T8_9, v(cm/s)]
    F = [25, x(1), x(2), x(3), x(4), 25];
    v_cms = x(5);

    [T_c, ~, ~] = solve_heat(F, v_cms, xm);
    T_s = T_c(T_c >= 30);

    if isempty(T_s)
        S = Inf; return;
    end

    m = analyze_curve(T_s);

    if ~check_constraints(m)
        S = Inf;  % 违反约束 → 无穷大惩罚
    else
        S = m.area;
    end
end

%% ==================== SelPSO 算法 ====================
function [xm_best, fv_best] = sel_pso(fitness, N, w, c1, c2, xmax, xmin, M, D, xm_data)
    % sel_pso - 带自然选择的粒子群优化算法
    %
    % 输入:
    %   fitness : 目标函数句柄 @(x, xm_data)
    %   N  : 粒子数
    %   w  : 惯性权重
    %   c1 : 个体学习因子
    %   c2 : 社会学习因子
    %   xmax, xmin : 变量上下界 [1×D]
    %   M  : 最大迭代次数
    %   D  : 变量维度
    %   xm_data : 传递给目标函数的额外数据 (热力学参数)

    % 速度限制
    Vmax = 0.2 * (xmax - xmin);

    % 初始化粒子位置和速度
    x = zeros(N, D); v = zeros(N, D);
    for i = 1:N
        x(i, :) = xmin + rand(1, D) .* (xmax - xmin);
        v(i, :) = Vmax .* (-1 + 2 * rand(1, D));
    end

    % 初始化个体最优和全局最优
    p = zeros(N, 1);  y = zeros(N, D);  % p:个体最优值, y:个体最优位置
    for i = 1:N
        p(i) = fitness(x(i, :), xm_data);
        y(i, :) = x(i, :);
    end

    % 找全局最优
    pg = p(N); px = x(N, :);
    for i = 1:N-1
        if p(i) < pg
            pg = p(i);
            px = x(i, :);
        end
    end

    % 适应度向量
    f = zeros(N, 1);

    % 主迭代循环
    for t = 1:M
        for i = 1:N
            % 速度更新
            v(i, :) = w * v(i, :) + c1 * rand(1, D) .* (y(i, :) - x(i, :)) ...
                                   + c2 * rand(1, D) .* (px - x(i, :));
            % 速度约束
            for j = 1:D
                if v(i, j) > Vmax(j),  v(i, j) = Vmax(j); end
                if v(i, j) < -Vmax(j), v(i, j) = -Vmax(j); end
            end

            % 位置更新
            x(i, :) = x(i, :) + v(i, :);

            % 边界检查 + 适应度评估
            if all(x(i, :) <= xmax) && all(x(i, :) >= xmin)
                f(i) = fitness(x(i, :), xm_data);
            else
                f(i) = Inf;
            end

            % 更新个体最优
            if f(i) < p(i)
                p(i) = f(i);
                y(i, :) = x(i, :);
            end
            % 更新全局最优
            if p(i) < pg
                pg = p(i);
                px = y(i, :);
            end
        end

        % ---- 自然选择: 用较好的粒子替换较差的粒子 ----
        [~, sort_idx] = sort(f);
        ex_index = round((N - 1) / 2);
        % 较差的粒子被较好的粒子替换 (保留位置和速度)
        x(sort_idx(N - ex_index + 1:N), :) = x(sort_idx(1:ex_index), :);
        v(sort_idx(N - ex_index + 1:N), :) = v(sort_idx(1:ex_index), :);

        % 显示进度
        if mod(t, 100) == 0
            fprintf('  迭代 %3d/%d, 当前最优面积: %.2f °C·s\n', t, M, pg);
        end
    end

    xm_best = px';
    fv_best = pg;
end

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
