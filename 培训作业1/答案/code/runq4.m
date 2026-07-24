% 单独运行Q4, 使用固定Q3参数作为基准
clc;

xm = [6.6857e-04, 2.1313e+04, ...
      8.1830e-04, 1.2193e+03, ...
      9.8977e-04, 7.2606e+02, ...
      8.6450e-04, 6.0728e+02, ...
      5.4317e-04, 9.9975e+02];

lb = [165, 185, 225, 245, 65/60];
ub = [185, 205, 245, 265, 100/60];

fprintf('=== Q4: 对称性优化 ===\n\n');

% Q3: 固定使用Table 6最优参数
x3_fixed = [178.1, 194.3, 225.9, 265.0, 90.90/60];
F3 = [25, x3_fixed(1), x3_fixed(2), x3_fixed(3), x3_fixed(4), 25];
[T3, ~, ~] = solve_heat(F3, x3_fixed(5), xm);
Ts3 = T3(T3 >= 30);
m3 = analyze_curve(Ts3);

% 计算Q3的对称性
dt = 0.5;
above = Ts3 - 217;
above = above(above >= 0);
[~, k] = max(above); n3 = length(above);
if k > 1
    AL3 = (sum(above(1:k)) - (above(1)+above(k))/2) * dt;
else
    AL3 = above(1)*dt/2;
end
if n3-k > 1
    AR3 = (sum(above(k:n3)) - (above(n3)+above(k))/2) * dt;
else
    AR3 = above(k)*dt/2;
end
s1_3 = abs(AL3-AR3)/max(abs(AL3), abs(AR3));
s2_3 = abs(n3+1-2*k)/max(k-1, n3-k);
sigma3 = max(s1_3, s2_3);

fprintf('Q3基准 (Table 6参数): area=%.1f, sigma=%.4f, Tmax=%.1f\n', m3.area, sigma3, m3.Tmax);
fprintf('  T1-5=%.1f, T6=%.1f, T7=%.1f, T8-9=%.1f, v=%.2f\n', ...
    x3_fixed(1), x3_fixed(2), x3_fixed(3), x3_fixed(4), x3_fixed(5)*60);

% Q4: 优化对称性
fprintf('\nQ4: SelPSO优化对称性...\n');
tic;
[x4, s4] = sel_pso(@obj_sym, 60, 0.8, 2.0, 2.0, ub, lb, 500, 5, xm);
toc;

F4 = [25, x4(1), x4(2), x4(3), x4(4), 25];
[T4, ~, ~] = solve_heat(F4, x4(5), xm);
Ts4 = T4(T4 >= 30);
m4 = analyze_curve(Ts4);

% 计算Q4对称性
above = Ts4 - 217;
above = above(above >= 0);
[~, k] = max(above); n4 = length(above);
if k > 1
    AL4 = (sum(above(1:k)) - (above(1)+above(k))/2) * dt;
else
    AL4 = above(1)*dt/2;
end
if n4-k > 1
    AR4 = (sum(above(k:n4)) - (above(n4)+above(k))/2) * dt;
else
    AR4 = above(k)*dt/2;
end
s1_4 = abs(AL4-AR4)/max(abs(AL4), abs(AR4));
s2_4 = abs(n4+1-2*k)/max(k-1, n4-k);
sigma4 = max(s1_4, s2_4);

fprintf('\n========================================\n');
fprintf('           面积       sigma    T1-5   T6    T7    T8-9  v(cm/min)\n');
fprintf('问题3: %8.1f   %7.4f  %5.1f %5.1f %5.1f %5.1f  %6.2f\n', ...
    m3.area, sigma3, x3_fixed(1), x3_fixed(2), x3_fixed(3), x3_fixed(4), x3_fixed(5)*60);
fprintf('问题4: %8.1f   %7.4f  %5.1f %5.1f %5.1f %5.1f  %6.2f\n', ...
    m4.area, sigma4, x4(1), x4(2), x4(3), x4(4), x4(5)*60);
fprintf('========================================\n');
fprintf('sigma1(面积)=%.4f, sigma2(时间)=%.4f, AL=%.1f, AR=%.1f\n', s1_4, s2_4, AL4, AR4);
fprintf('\nQ4约束: Tmax=%.1f, slope=%.3f, t150-190=%.1f, t>217=%.1f, ok=%d\n', ...
    m4.Tmax, m4.max_slope, m4.t_150_190, m4.t_above, check_constraints(m4));
fprintf('Delta_area=%.1f (%.1f%%), Delta_sigma=%.4f\n', ...
    m4.area-m3.area, (m4.area-m3.area)/m3.area*100, sigma4-sigma3);

fprintf('\nDone.\n');

% =============== 目标函数 ===============
function sigma = obj_sym(x, xm)
    F = [25, x(1), x(2), x(3), x(4), 25];
    [T_c, ~, ~] = solve_heat(F, x(5), xm);
    T_s = T_c(T_c >= 30);
    if isempty(T_s), sigma = Inf; return; end
    m = analyze_curve(T_s);
    if ~check_constraints(m), sigma = Inf; else
        dt = 0.5;
        above = T_s - 217;
        above = above(above >= 0);
        [~, k] = max(above); n = length(above);
        if k > 1, AL = (sum(above(1:k)) - (above(1)+above(k))/2) * dt; else, AL = above(1)*dt/2; end
        if n-k > 1, AR = (sum(above(k:n)) - (above(n)+above(k))/2) * dt; else, AR = above(k)*dt/2; end
        s1 = abs(AL-AR)/max(abs(AL), abs(AR));
        s2 = abs(n+1-2*k)/max(k-1, n-k);
        sigma = max(s1, s2);
    end
end

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
