%% 参数估计: PSO 拟合热力学参数
% 利用附件实验数据 (v=70cm/min)
% 待估10参数: [a1,h1, a2,h2, a3,h3, a4,h4, a5,h5]

clc; clear;

% 加载实验数据
E = readmatrix('..\题目\附件.xlsx');
t_exp = E(:,1); T_exp = E(:,2);

F_exp = [25, 175, 195, 235, 255, 25];
v_exp = 70/60;

% 变量范围
lb = [4e-4, 20000, 4e-4, 700, 4e-4, 700, 4e-4, 200, 4e-4, 700];
ub = [9e-4, 30000, 2e-3, 1500, 2e-3, 1500, 2e-3, 1000, 2e-3, 1500];

fprintf('参数估计: PSO拟合 (60粒子, 1000代)\n');
fprintf('实验数据: %d点, v=70cm/min\n', length(T_exp));

tic;
[xm_opt, SSE] = sel_pso(@obj_fit, 60, 0.8, 2.05, 2.05, ub, lb, 1000, 10, ...
                         F_exp, v_exp, t_exp, T_exp);
toc;

% 验证
[T_sim, ~, ~] = solve_heat(F_exp, v_exp, xm_opt);
idx0 = find(T_sim >= 30, 1);
T_sim_s = T_sim(idx0:end);
t_sim = ((idx0-1) + (0:length(T_sim_s)-1)) * 0.5;
T_interp = interp1(t_sim, T_sim_s, t_exp, 'linear');
err = T_interp - T_exp;
R2 = 1 - sum(err.^2)/sum((T_exp-mean(T_exp)).^2);

fprintf('\n拟合结果 (10参数):\n');
fprintf('区域              a           h\n');
fprintf('预热区(1-5):  %.4e  %.4e\n', xm_opt(1), xm_opt(2));
fprintf('恒温区(6):    %.4e  %.4e\n', xm_opt(3), xm_opt(4));
fprintf('回流升温(7):  %.4e  %.4e\n', xm_opt(5), xm_opt(6));
fprintf('回流峰值(8-9):%.4e  %.4e\n', xm_opt(7), xm_opt(8));
fprintf('冷却区(10-11):%.4e  %.4e\n', xm_opt(9), xm_opt(10));
fprintf('\nSSE=%.4f, RMSE=%.4f°C, R²=%.4f\n', SSE, sqrt(mean(err.^2)), R2);

% 绘图
figure('Position', [100, 100, 1100, 420]);
subplot(1,2,1); hold on;
plot(t_exp, T_exp, 'b.', 'MarkerSize', 6);
plot(t_exp, T_interp, 'r-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('温度 (°C)');
title(sprintf('拟合对比 (R^2=%.4f, RMSE=%.2f°C)', R2, sqrt(mean(err.^2))));
legend('实验数据', '模拟值', 'Location', 'southeast'); grid on;

subplot(1,2,2); hold on;
stem(t_exp, err, 'k.', 'MarkerSize', 4);
yline(0, 'r-', 'LineWidth', 1);
yline(3*sqrt(mean(err.^2)), 'b--', 'LineWidth', 1);
yline(-3*sqrt(mean(err.^2)), 'b--', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('残差 (°C)');
title(sprintf('残差分析 (RMSE=%.2f°C)', sqrt(mean(err.^2))));
legend('残差', '零线', '±3RMSE', 'Location', 'southeast'); grid on;

print(gcf, 'parameter_fit.png', '-dpng', '-r200');
save('fitted_parameters.mat', 'xm_opt', 'SSE', 'R2');

% =============== 目标函数 ===============
function sse = obj_fit(xm, F_exp, v_exp, t_exp, T_exp)
    [T_sim, ~, ~] = solve_heat(F_exp, v_exp, xm);
    idx0 = find(T_sim >= 30, 1);
    if isempty(idx0), sse = Inf; return; end
    T_s = T_sim(idx0:end);
    t_s = ((idx0-1) + (0:length(T_s)-1)) * 0.5;  % 传感器启动时刻对齐
    T_int = interp1(t_s, T_s, t_exp, 'linear');
    if any(isnan(T_int)), sse = Inf; else, sse = sum((T_int-T_exp).^2); end
end

% =============== SelPSO (10维) ===============
function [xm_best, fv_best] = sel_pso(fit, N, w, c1, c2, xmax, xmin, M, D, ...
                                      F_exp, v_exp, t_exp, T_exp)
    Vmax = 0.2*(xmax - xmin);
    x = xmin + rand(N, D).*(xmax - xmin);
    v = Vmax.*(-1 + 2*rand(N, D));
    p = zeros(N,1); y = x;
    for i = 1:N, p(i) = fit(x(i,:), F_exp, v_exp, t_exp, T_exp); end
    [pg, idx] = min(p); px = x(idx,:);

    f = zeros(N,1);
    for t = 1:M
        for i = 1:N
            v(i,:) = w*v(i,:) + c1*rand(1,D).*(y(i,:)-x(i,:)) + c2*rand(1,D).*(px-x(i,:));
            v(i,:) = max(min(v(i,:), Vmax), -Vmax);
            x(i,:) = x(i,:) + v(i,:);
            if all(x(i,:) <= xmax) && all(x(i,:) >= xmin)
                f(i) = fit(x(i,:), F_exp, v_exp, t_exp, T_exp);
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
        if mod(t,200)==0, fprintf('  迭代%3d: SSE=%.4f\n', t, pg); end
    end
    xm_best = px'; fv_best = pg;
end
