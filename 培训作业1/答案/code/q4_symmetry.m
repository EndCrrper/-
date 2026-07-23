%% 问题4: 对称性优化
% 在制程界限下, 最小化对称性指标 sigma
% sigma = max(|AL-AR|/max(AL,AR), |dtR-dtL|/max(dtL,dtR))

clc; clear;
addpath(fileparts(mfilename('fullpath')));

xm = [6.6857e-04, 2.1313e+04, ...
      8.1830e-04, 1.2193e+03, ...
      9.8977e-04, 7.2606e+02, ...
      8.6450e-04, 6.0728e+02, ...
      5.4317e-04, 9.9975e+02];

lb = [165, 185, 225, 245, 65/60];
ub = [185, 205, 245, 265, 100/60];

fprintf('问题4: 对称性优化 (SelPSO)\n\n');

% Q4: 最小化对称性
tic;
[x4, s4] = sel_pso(@obj_sym, 60, 0.8, 2.0, 2.0, ub, lb, 500, 5, xm);
toc;

F4 = [25, x4(1), x4(2), x4(3), x4(4), 25];
[T4, T_oven4, ~] = solve_heat(F4, x4(5), xm);
T_s4 = T4(T4 >= 30);
m4 = analyze_curve(T_s4);
[sigma4, s1, s2, AL, AR] = calc_sym(T_s4);

% Q3: 最小化面积 (对比基准)
fprintf('\n运行Q3对比...\n');
[x3, S3] = sel_pso(@obj_area_q4, 60, 0.8, 2.0, 2.0, ub, lb, 500, 5, xm);
F3 = [25, x3(1), x3(2), x3(3), x3(4), 25];
[T3, ~, ~] = solve_heat(F3, x3(5), xm);
T_s3 = T3(T3 >= 30);
m3 = analyze_curve(T_s3);
sigma3 = calc_sym(T_s3);

fprintf('\n========================================\n');
fprintf('           面积       sigma    T1-5   T6    T7    T8-9  v(cm/min)\n');
fprintf('问题3: %8.1f   %7.4f  %5.1f %5.1f %5.1f %5.1f  %6.2f\n', ...
    S3, sigma3, x3(1), x3(2), x3(3), x3(4), x3(5)*60);
fprintf('问题4: %8.1f   %7.4f  %5.1f %5.1f %5.1f %5.1f  %6.2f\n', ...
    m4.area, sigma4, x4(1), x4(2), x4(3), x4(4), x4(5)*60);
fprintf('========================================\n');
fprintf('sigma1(面积)=%.4f, sigma2(时间)=%.4f, AL=%.1f, AR=%.1f\n', s1, s2, AL, AR);

% 绘图
L_total = 25 + 11*30.5 + 10*5 + 25;
total_t = L_total / x4(5);
t_full = (0:length(T4)-1) * 0.5;
x_full = x4(5) * t_full;
[~, pk] = max(T4);

figure('Position', [100, 100, 1200, 450]);
subplot(1,2,1); hold on;
h1 = plot(t_full, T4, 'r-', 'LineWidth', 1.5);
h2 = plot(t_full, T_oven4, 'b--', 'LineWidth', 1);
h3 = yline(217, 'g--', 'LineWidth', 1);
h4 = xline(t_full(pk), 'k:', 'LineWidth', 1);
h5 = plot(t_full(pk), T4(pk), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold off; xlabel('时间 (s)'); ylabel('温度 (°C)');
title(sprintf('问题4: 对称最优 (sigma=%.4f, 面积=%.1f °C·s)', sigma4, m4.area));
legend([h1, h2, h3, h4, h5], {'焊接中心温度', '炉内环境温度', 'T=217°C', '峰值对称线', '峰值'}, ...
    'Location', 'southeast', 'FontSize', 8);
grid on;

% 对称性镜像对比
subplot(1,2,2); hold on;
idx_s = find(T4 >= 30, 1);
T_post = T4(idx_s:end); t_post = t_full(idx_s:end);
[~, pk2] = max(T_post);
t_L = t_post(1:pk2) - t_post(pk2);
t_R = t_post(pk2:end) - t_post(pk2);
h1 = plot(t_L, T_post(1:pk2), 'r-', 'LineWidth', 1.5);
h2 = plot(-t_R, T_post(pk2:end), 'b--', 'LineWidth', 1.5);
h3 = yline(217, 'g--', 'LineWidth', 1);
hold off; xlabel('相对峰值时间 (s)'); ylabel('温度 (°C)');
title(sprintf('对称性分析 (sigma1=%.4f, sigma2=%.4f)', s1, s2));
legend([h1, h2, h3], {'峰值左侧', '峰值右侧(镜像)', 'T=217°C'}, ...
    'Location', 'southeast', 'FontSize', 8);
grid on;

print(gcf, fullfile(fileparts(mfilename('fullpath')), '..', 'result', 'figures', 'q4_symmetric_optimal.png'), '-dpng', '-r300');

% =============== 对称性计算 ===============
function [sigma, s1, s2, AL, AR] = calc_sym(T)
    dt = 0.5;
    above = T - 217;
    above = above(above >= 0);
    if isempty(above), sigma = 1; s1 = 1; s2 = 1; AL = 0; AR = 0; return; end
    [~, k] = max(above); n = length(above);

    if k > 1
        AL = (sum(above(1:k)) - (above(1)+above(k))/2) * dt;
    else
        AL = above(1)*dt/2;
    end
    if n-k > 1
        AR = (sum(above(k:n)) - (above(n)+above(k))/2) * dt;
    else
        AR = above(k)*dt/2;
    end

    s1 = abs(AL-AR)/max(abs(AL), abs(AR));
    s2 = abs(n+1-2*k)/max(k-1, n-k);
    sigma = max(s1, s2);
end

% =============== 目标函数 ===============
function sigma = obj_sym(x, xm)
    F = [25, x(1), x(2), x(3), x(4), 25];
    [T_c, ~, ~] = solve_heat(F, x(5), xm);
    T_s = T_c(T_c >= 30);
    if isempty(T_s), sigma = Inf; return; end
    m = analyze_curve(T_s);
    if ~check_constraints(m), sigma = Inf; else, sigma = calc_sym(T_s); end
end

function S = obj_area_q4(x, xm)
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
