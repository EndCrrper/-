%% ============================================================
% 问题4: 带对称性要求的炉温曲线优化
% ============================================================
% 在问题3基础上，进一步要求峰值温度两侧超过217°C的曲线尽量对称
%
% 对称性指标: σ = max(σ_area, σ_time)
%   σ_area = |A_L - A_R| / max(A_L, A_R)     (面积对称性)
%   σ_time = |ΔT_R - ΔT_L| / max(ΔT_L, ΔT_R) (时间对称性)
%
% 优化方法: SelPSO, 目标函数嵌入对称性指标
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
lb = [165, 185, 225, 245, 65/60];
ub = [185, 205, 245, 265, 100/60];

% PSO 参数
N = 60; w = 0.8; c1 = 2.0; c2 = 2.0;
M = 500; D = 5;

fprintf('========== 问题4: 对称性优化 ==========\n');
fprintf('优化方法: SelPSO (目标函数=对称性指标σ)\n');

%% ---- SelPSO 优化 (直接最小化对称性) ----
tic;
[best_x, best_sigma] = sel_pso_sym(@objective_symmetry, N, w, c1, c2, ub, lb, M, D, xm);
elapsed = toc;

fprintf('\n优化完成! 耗时: %.2f 秒\n', elapsed);

%% ---- 结果输出 ----
F_opt = [25, best_x(1), best_x(2), best_x(3), best_x(4), 25];
[T_c, ~, T_oven] = solve_heat(F_opt, best_x(5), xm);
T_s = T_c(T_c >= 30);
m = analyze_curve(T_s);
[sigma_actual, sigma1, sigma2, AL, AR] = compute_symmetry(T_s);

fprintf('\n========== 问题4 最优解 ==========\n');
fprintf('  小温区1~5温度:  %.1f °C\n', best_x(1));
fprintf('  小温区6温度:    %.1f °C\n', best_x(2));
fprintf('  小温区7温度:    %.1f °C\n', best_x(3));
fprintf('  小温区8~9温度:  %.1f °C\n', best_x(4));
fprintf('  传送带速度:     %.2f cm/min\n', best_x(5)*60);
fprintf('\n炉温曲线指标:\n');
fprintf('  面积:           %.1f °C·s\n', m.area);
fprintf('  对称性 σ:       %.4f\n', sigma_actual);
fprintf('    σ₁ (面积):    %.4f  (A_L=%.1f, A_R=%.1f)\n', sigma1, AL, AR);
fprintf('    σ₂ (时间):    %.4f\n', sigma2);
fprintf('  峰值温度:       %.1f °C\n', m.Tmax);
fprintf('  最大速率:       %.3f °C/s\n', m.max_slope);
fprintf('  150-190°C时间:  %.1f s\n', m.t_150_190);
fprintf('  >217°C时间:     %.1f s\n', m.t_above);

fprintf('\n--- 约束验证 ---\n');
fprintf('  所有约束满足: %s\n', ternary(check_constraints(m), '✓', '✗'));

%% ---- 与问题3的对比 ----
% 运行问题3优化获取基准面积
fprintf('\n========== 问题3 vs 问题4 对比 ==========\n');

% 简化版Q3结果 (如果已运行q3，可加载工作区变量)
[x_q3, fval_q3] = sel_pso_sym(@objective_area_only, N, w, c1, c2, ub, lb, M, D, xm);
F_q3 = [25, x_q3(1), x_q3(2), x_q3(3), x_q3(4), 25];
[T_q3, ~, ~] = solve_heat(F_q3, x_q3(5), xm);
T_s_q3 = T_q3(T_q3 >= 30);
m_q3 = analyze_curve(T_s_q3);
sigma_q3 = compute_symmetry(T_s_q3);

fprintf('            |   面积(°C·s) |   对称性σ | T1-5 |  T6  |  T7  | T8-9 | v(cm/min)\n');
fprintf('  ----------|--------------|-----------|------|------|------|------|----------\n');
fprintf('  问题3     |  %8.1f    |  %7.4f  |%5.1f|%5.1f|%5.1f|%5.1f|  %6.2f\n', ...
    fval_q3, sigma_q3, x_q3(1), x_q3(2), x_q3(3), x_q3(4), x_q3(5)*60);
fprintf('  问题4     |  %8.1f    |  %7.4f  |%5.1f|%5.1f|%5.1f|%5.1f|  %6.2f\n', ...
    m.area, sigma_actual, best_x(1), best_x(2), best_x(3), best_x(4), best_x(5)*60);

%% ---- 绘图 ----
L_total = 25 + 11*30.5 + 10*5 + 25;
total_time = L_total / best_x(5);
dt = 0.5;
t_full = (0:length(T_c)-1) * dt;

figure('Position', [100, 100, 1400, 500]);

% 子图1: 最优对称炉温曲线
subplot(1,2,1);
hold on;
[~, peak_idx] = max(T_c);
plot(t_full, T_c, 'r-', 'LineWidth', 1.5);
plot(t_full, T_oven, 'b--', 'LineWidth', 1);
yline(217, 'g--', 'LineWidth', 0.8);
plot(t_full(peak_idx), T_c(peak_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
% 标注峰值对称线
xline(t_full(peak_idx), 'k:', 'LineWidth', 1);
hold off;
xlabel('时间 (s)'); ylabel('温度 (°C)');
title(sprintf('问题4: 对称最优炉温曲线 (σ=%.4f, 面积=%.1f)', sigma_actual, m.area));
legend('焊接区域中心温度', '炉内环境温度', 'T=217°C', '峰值', '峰值对称线');
grid on;

% 子图2: 对称性分析 (峰值两侧镜像对比)
subplot(1,2,2);
idx_s = find(T_c >= 30, 1, 'first');
if isempty(idx_s), idx_s = 1; end
T_post = T_c(idx_s:end);
t_post = t_full(idx_s:end);
[~, pk_local] = max(T_post);

t_left = t_post(1:pk_local) - t_post(pk_local);
t_right = t_post(pk_local:end) - t_post(pk_local);

hold on;
plot(t_left, T_post(1:pk_local), 'r-', 'LineWidth', 1.5);
plot(-t_right, T_post(pk_local:end), 'b--', 'LineWidth', 1.5);
yline(217, 'g--', 'LineWidth', 0.8);
hold off;
xlabel('相对峰值时间 (s)'); ylabel('温度 (°C)');
title(sprintf('对称性分析 (σ=%.4f, σ₁=%.4f, σ₂=%.4f)', sigma_actual, sigma1, sigma2));
legend('峰值左侧', '峰值右侧(镜像)', 'T=217°C');
grid on;

saveas(gcf, 'q4_symmetric_optimal.png');
fprintf('\n图已保存: q4_symmetric_optimal.png\n');
fprintf('问题4 求解完成!\n');

%% ==================== 对称性计算函数 ====================
function [sigma, sigma1, sigma2, AL, AR] = compute_symmetry(T)
    % 计算炉温曲线的对称性指标
    % σ = max(σ₁, σ₂)
    % σ₁: 面积对称性  σ₂: 时间对称性

    dt = 0.5;
    above = T - 217;
    above = above(above > 0);

    if isempty(above)
        sigma = 1.0; sigma1 = 1.0; sigma2 = 1.0;
        AL = 0; AR = 0; return;
    end

    [~, k] = max(above);  % 峰值在above序列中的位置
    n_above = length(above);

    % 左侧面积 (峰值之前)
    if k > 2
        AL = (sum(above(1:k)) - (above(1) + above(k)) / 2) * dt;
    elseif k > 1
        AL = (above(1) + above(k)) * dt / 2;
    else
        AL = above(k) * dt / 2;
    end

    % 右侧面积 (峰值之后)
    if n_above - k > 2
        AR = (sum(above(k:n_above)) - (above(n_above) + above(k)) / 2) * dt;
    elseif n_above - k > 1
        AR = (above(n_above) + above(k)) * dt / 2;
    else
        AR = above(k) * dt / 2;
    end

    % σ₁: 面积对称性
    sigma1 = abs(AL - AR) / max(abs(AL), abs(AR));

    % σ₂: 时间对称性
    sigma2 = abs(n_above + 1 - 2*k) / max(k - 1, n_above - k);

    % 综合对称性
    sigma = max(sigma1, sigma2);
end

%% ==================== 目标函数 ====================
function sigma = objective_symmetry(x, xm)
    % Q4目标: 最小化对称性指标 σ
    F = [25, x(1), x(2), x(3), x(4), 25];
    v_cms = x(5);

    [T_c, ~, ~] = solve_heat(F, v_cms, xm);
    T_s = T_c(T_c >= 30);

    if isempty(T_s)
        sigma = Inf; return;
    end

    m = analyze_curve(T_s);

    if ~check_constraints(m)
        sigma = Inf;
    else
        sigma = compute_symmetry(T_s);
    end
end

function S = objective_area_only(x, xm)
    % Q3目标: 最小化面积 (用于对比)
    F = [25, x(1), x(2), x(3), x(4), 25];
    v_cms = x(5);

    [T_c, ~, ~] = solve_heat(F, v_cms, xm);
    T_s = T_c(T_c >= 30);

    if isempty(T_s)
        S = Inf; return;
    end

    m = analyze_curve(T_s);

    if ~check_constraints(m)
        S = Inf;
    else
        S = m.area;
    end
end

%% ==================== SelPSO (对称性版本) ====================
function [xm_best, fv_best] = sel_pso_sym(fitness, N, w, c1, c2, xmax, xmin, M, D, xm_data)
    % 带自然选择的粒子群优化 (与Q3中相同算法)

    Vmax = 0.2 * (xmax - xmin);

    x = zeros(N, D); v = zeros(N, D);
    for i = 1:N
        x(i, :) = xmin + rand(1, D) .* (xmax - xmin);
        v(i, :) = Vmax .* (-1 + 2 * rand(1, D));
    end

    p = zeros(N, 1); y = zeros(N, D);
    for i = 1:N
        p(i) = fitness(x(i, :), xm_data);
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
                f(i) = fitness(x(i, :), xm_data);
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
        [~, sort_idx] = sort(f);
        ex_index = round((N - 1) / 2);
        x(sort_idx(N - ex_index + 1:N), :) = x(sort_idx(1:ex_index), :);
        v(sort_idx(N - ex_index + 1:N), :) = v(sort_idx(1:ex_index), :);

        if mod(t, 100) == 0
            fprintf('  迭代 %3d/%d, 当前最优 σ: %.4f\n', t, M, pg);
        end
    end

    xm_best = px';
    fv_best = pg;
end

%% ---- 辅助函数 ----
function s = ternary(cond, t, f)
    if cond, s = t; else, s = f; end
end
