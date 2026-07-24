%% 生成 PSO 收敛曲线图
clc; clear;
addpath(fileparts(mfilename('fullpath')));

xm = [6.6857e-04, 2.1313e+04, 8.1830e-04, 1.2193e+03, ...
      9.8977e-04, 7.2606e+02, 8.6450e-04, 6.0728e+02, ...
      5.4317e-04, 9.9975e+02];

lb = [165, 185, 225, 245, 65/60];
ub = [185, 205, 245, 265, 100/60];
N = 60; w = 0.8; c1 = 2.0; c2 = 2.0; M = 500; D = 5;

fprintf('生成 PSO 收敛曲线...\n');

% --- Q3 收敛曲线 (最小化面积) ---
[~, ~, history3] = sel_pso_log(@obj_area_q3, N, w, c1, c2, ub, lb, M, D, xm);

% --- Q4 收敛曲线 (最小化对称性) ---
[~, ~, history4] = sel_pso_log(@obj_sym_q4, N, w, c1, c2, ub, lb, M, D, xm);

% --- 绘图 (双 y 轴线性) ---
figure('Position', [100, 100, 1100, 420]);
yyaxis left;
plot(0:M, history3, 'b-', 'LineWidth', 1.2);
ylabel('面积 S (°C·s)', 'FontSize', 11);
ylim([380, 520]);

yyaxis right;
plot(0:M, history4, 'r-', 'LineWidth', 1.2);
ylabel('对称性指标 \sigma', 'FontSize', 11);
ylim([0.16, 0.38]);

xlabel('迭代次数', 'FontSize', 11);
title('SelPSO 收敛曲线 (60粒子 \times 500迭代)', 'FontSize', 12);
legend('问题3: 最小化面积 (左轴)', '问题4: 最小化对称性 \sigma (右轴)', ...
    'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

rootDir = fullfile(fileparts(mfilename('fullpath')), '..');
print(gcf, fullfile(rootDir, 'result', 'figures', 'pso_convergence.png'), '-dpng', '-r300');
fprintf('已保存: pso_convergence.png\n');

% ========== SelPSO with convergence log ==========
function [xm_best, fv_best, history] = sel_pso_log(fit, N, w, c1, c2, xmax, xmin, M, D, xm_data)
    Vmax = 0.2*(xmax - xmin);
    x = xmin + rand(N,D).*(xmax - xmin);
    v = Vmax.*(-1 + 2*rand(N,D));
    p = zeros(N,1); y = x;
    for i = 1:N, p(i) = fit(x(i,:), xm_data); end
    [pg, idx] = min(p); px = x(idx,:);
    history = zeros(M+1, 1);
    history(1) = pg;

    f = zeros(N,1);
    for t = 1:M
        for i = 1:N
            v(i,:) = w*v(i,:) + c1*rand(1,D).*(y(i,:)-x(i,:)) + c2*rand(1,D).*(px-x(i,:));
            v(i,:) = max(min(v(i,:), Vmax), -Vmax);
            x(i,:) = x(i,:) + v(i,:);
            if all(x(i,:)<=xmax) && all(x(i,:)>=xmin)
                f(i) = fit(x(i,:), xm_data);
            else
                f(i) = Inf;
            end
            if f(i) < p(i), p(i)=f(i); y(i,:)=x(i,:); end
            if p(i) < pg, pg=p(i); px=y(i,:); end
        end
        [~, s_idx] = sort(f);
        ex = round((N-1)/2);
        x(s_idx(N-ex+1:N),:) = x(s_idx(1:ex),:);
        v(s_idx(N-ex+1:N),:) = v(s_idx(1:ex),:);
        history(t+1) = pg;
    end
    xm_best = px'; fv_best = pg;
end

function S = obj_area_q3(x, xm)
    F = [25, x(1), x(2), x(3), x(4), 25];
    [T_c,~,~] = solve_heat(F, x(5), xm);
    T_s = T_c(T_c>=30);
    if isempty(T_s), S = Inf; return; end
    m = analyze_curve(T_s);
    if check_constraints(m), S = m.area; else, S = Inf; end
end

function sigma = obj_sym_q4(x, xm)
    F = [25, x(1), x(2), x(3), x(4), 25];
    [T_c,~,~] = solve_heat(F, x(5), xm);
    T_s = T_c(T_c>=30);
    if isempty(T_s), sigma = Inf; return; end
    m = analyze_curve(T_s);
    if ~check_constraints(m), sigma = Inf; return; end
    above = T_s - 217; above = above(above>=0);
    if isempty(above), sigma = 1; return; end
    [~, k] = max(above); n = length(above);
    if k>1, AL=(sum(above(1:k))-(above(1)+above(k))/2)*0.5; else, AL=above(1)*0.5/2; end
    if n-k>1, AR=(sum(above(k:n))-(above(n)+above(k))/2)*0.5; else, AR=above(k)*0.5/2; end
    s1 = abs(AL-AR)/max(abs(AL),abs(AR));
    s2 = abs(n+1-2*k)/max(k-1,n-k);
    sigma = max(s1,s2);
end
