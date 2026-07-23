%% 问题3: 最小化超过217°C到峰值的面积
% SelPSO 优化
% 变量: [T1_5, T6, T7, T8_9, v(cm/s)]
% 目标: min S (面积)
% 约束: 制程界限 + 变量范围

clc; clear;

xm = [6.6857e-04, 2.1313e+04, ...
      8.1830e-04, 1.2193e+03, ...
      9.8977e-04, 7.2606e+02, ...
      8.6450e-04, 6.0728e+02, ...
      5.4317e-04, 9.9975e+02];

% 变量范围
lb = [165, 185, 225, 245, 65/60];
ub = [185, 205, 245, 265, 100/60];

fprintf('问题3: 最小化面积 (SelPSO, 60粒子×500代)\n\n');

tic;
[x_opt, S_min] = sel_pso(@obj_area, 60, 0.8, 2.0, 2.0, ub, lb, 500, 5, xm);
toc;

F_opt = [25, x_opt(1), x_opt(2), x_opt(3), x_opt(4), 25];
[T_c, T_oven, ~] = solve_heat(F_opt, x_opt(5), xm);
T_s = T_c(T_c >= 30);
m = analyze_curve(T_s);

fprintf('\n最优解:\n');
fprintf('  T1-5 = %.1f°C\n', x_opt(1));
fprintf('  T6   = %.1f°C\n', x_opt(2));
fprintf('  T7   = %.1f°C\n', x_opt(3));
fprintf('  T8-9 = %.1f°C\n', x_opt(4));
fprintf('  v    = %.2f cm/min\n', x_opt(5)*60);
fprintf('  面积 = %.1f °C·s\n', S_min);
fprintf('  Tmax = %.1f°C, 斜率=%.3f°C/s, 150-190=%.1fs, >217=%.1fs\n', ...
    m.Tmax, m.max_slope, m.t_150_190, m.t_above);

% 绘图
L_total = 25 + 11*30.5 + 10*5 + 25;
total_t = L_total / x_opt(5);
t_full = (0:length(T_c)-1) * 0.5;
x_full = x_opt(5) * t_full;
[~, pk] = max(T_c);
idx_217 = find(T_c >= 217, 1);

figure('Position', [100, 100, 1200, 450]);
subplot(1,2,1); hold on;
if ~isempty(idx_217)
    x_fill = t_full(idx_217:pk);
    y_top = T_c(idx_217:pk);
    y_bot = 217 * ones(size(x_fill));
    h_fill = fill([x_fill, fliplr(x_fill)], [y_top(:)', y_bot(:)'], ...
        [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end
h1 = plot(t_full, T_c, 'r-', 'LineWidth', 1.5);
h2 = plot(t_full, T_oven, 'b--', 'LineWidth', 1);
h3 = yline(217, 'g--', 'LineWidth', 1);
h4 = plot(t_full(pk), T_c(pk), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold off; xlabel('时间 (s)'); ylabel('温度 (°C)');
title(sprintf('问题3: 最优炉温曲线 (面积=%.1f °C·s)', S_min));
if exist('h_fill', 'var')
    legend([h_fill, h1, h2, h3, h4], {'>217°C至峰值面积', '焊接中心温度', '炉内环境温度', 'T=217°C', '峰值'}, ...
        'Location', 'southeast', 'FontSize', 8);
end
grid on;

subplot(1,2,2); hold on;
h1 = plot(x_full, T_c, 'r-', 'LineWidth', 1.5);
h2 = plot(x_full, T_oven, 'b--', 'LineWidth', 1);
h3 = yline(217, 'g--', 'LineWidth', 1);
h4 = plot(x_full(pk), T_c(pk), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold off; xlabel('位置 (cm)'); ylabel('温度 (°C)');
title('问题3: 位置-温度曲线');
legend([h1, h2, h3, h4], {'焊接中心温度', '炉内环境温度', 'T=217°C', '峰值'}, ...
    'Location', 'southeast', 'FontSize', 8);
grid on;

print(gcf, 'q3_optimal.png', '-dpng', '-r200');

% =============== 目标函数 ===============
function S = obj_area(x, xm)
    F = [25, x(1), x(2), x(3), x(4), 25];
    [T_c, ~, ~] = solve_heat(F, x(5), xm);
    T_s = T_c(T_c >= 30);
    if isempty(T_s), S = Inf; return; end
    m = analyze_curve(T_s);
    if check_constraints(m), S = m.area; else, S = Inf; end
end

% =============== SelPSO ===============
function [xm_best, fv_best] = sel_pso(fit, N, w, c1, c2, xmax, xmin, M, D, xm_data)
    Vmax = 0.2*(xmax - xmin);
    x = xmin + rand(N, D).*(xmax - xmin);
    v = Vmax.*(-1 + 2*rand(N, D));
    p = zeros(N,1); y = x;
    for i = 1:N, p(i) = fit(x(i,:), xm_data); end
    [pg, idx] = min(p); px = x(idx,:);

    f = zeros(N,1);
    for t = 1:M
        for i = 1:N
            v(i,:) = w*v(i,:) + c1*rand(1,D).*(y(i,:)-x(i,:)) + c2*rand(1,D).*(px-x(i,:));
            v(i,:) = max(min(v(i,:), Vmax), -Vmax);
            x(i,:) = x(i,:) + v(i,:);
            if all(x(i,:) <= xmax) && all(x(i,:) >= xmin)
                f(i) = fit(x(i,:), xm_data);
            else
                f(i) = Inf;
            end
            if f(i) < p(i), p(i) = f(i); y(i,:) = x(i,:); end
            if p(i) < pg, pg = p(i); px = y(i,:); end
        end
        [~, s_idx] = sort(f);
        ex = round((N-1)/2);
        x(s_idx(N-ex+1:N),:) = x(s_idx(1:ex),:);
        v(s_idx(N-ex+1:N),:) = v(s_idx(1:ex),:);
    end
    xm_best = px'; fv_best = pg;
end
