function [T_center, T_oven, t_full] = solve_heat(F, v_cms, xm)
% 一维热传导 Crank-Nicolson 求解器
% F: 各段温度 [T_front, T1_5, T6, T7, T8_9, T_rear]
% v_cms: 传送带速度 (cm/s)
% xm: 10参数 [a1,h1, a2,h2, a3,h3, a4,h4, a5,h5]

    L_total = 25 + 11*30.5 + 10*5 + 25;  % 435.5 cm
    thickness = 0.015;
    dx = 1e-4;
    dt = 0.5;
    T_ambient = 25;

    total_time = L_total / v_cms;
    n = ceil(thickness / dx) + 1;
    m = floor(total_time / dt) + 1;
    k_center = ceil(thickness / 2 / dx);

    % 炉内环境温度序列
    t_vec = (0:m-1) * dt;
    x_vec = v_cms * t_vec;
    T_oven = oven_temp(x_vec, F);

    % 各段分界
    L1 = 25 + 5*30.5 + 5*5;    % 197.5
    L2 = L1 + 30.5 + 5;         % 233
    L3 = L2 + 30.5 + 5;         % 268.5
    t1 = L1/v_cms; t2 = L2/v_cms; t3 = L3/v_cms;
    m1 = floor(t1/dt) + 1;
    m2 = floor(t2/dt) + 1;
    m3 = floor(t3/dt) + 1;

    % 热力学参数: r = a^2 * dt / dx^2
    a_vals = [xm(1), xm(3), xm(5), xm(7), xm(9)];
    h_vals = [xm(2), xm(4), xm(6), xm(8), xm(10)];
    r_vals = a_vals.^2 * dt / (dx^2);

    % 温度场初始化
    u = zeros(n, m);
    u(:, 1) = T_ambient;
    T_center = ones(m, 1) * T_ambient;

    % 段1: 炉前 + 温区1~5
    [C1, d1] = build_cn(n, r_vals(1), h_vals(1), m, T_oven, dx);
    for j = 1:m1-1
        u(:, j+1) = C1 * u(:, j) + d1(:, j+1);
        T_center(j+1) = u(k_center, j+1);
    end

    % 段2: 温区6
    [C2, d2] = build_cn(n, r_vals(2), h_vals(2), m, T_oven, dx);
    for j = m1:m2-1
        u(:, j+1) = C2 * u(:, j) + d2(:, j+1);
        T_center(j+1) = u(k_center, j+1);
    end

    % 段3: 温区7
    [C3, d3] = build_cn(n, r_vals(3), h_vals(3), m, T_oven, dx);
    for j = m2:m3-1
        u(:, j+1) = C3 * u(:, j) + d3(:, j+1);
        T_center(j+1) = u(k_center, j+1);
    end

    % 段4&5: 升温/降温自动切换
    [C4, d4] = build_cn(n, r_vals(4), h_vals(4), m, T_oven, dx);
    [C5, d5] = build_cn(n, r_vals(5), h_vals(5), m, T_oven, dx);
    for j = m3:m-1
        if T_center(j) >= T_center(j-1)
            u(:, j+1) = C4 * u(:, j) + d4(:, j+1);
        else
            u(:, j+1) = C5 * u(:, j) + d5(:, j+1);
        end
        T_center(j+1) = u(k_center, j+1);
    end

    if nargout >= 3
        t_full = (0:m-1) * dt;
    end
end

function [C, d] = build_cn(n, r, h, m, T_oven, dx)
    A = zeros(n, n);
    B = zeros(n, n);

    for i = 2:n-1
        A(i,i) = 2*(1+r); A(i,i-1) = -r; A(i,i+1) = -r;
        B(i,i) = 2*(1-r); B(i,i-1) =  r; B(i,i+1) =  r;
    end

    A(1,1) = 1 + h*dx; A(1,2) = -1; B(1,:) = 0;
    A(n,n) = 1 + h*dx; A(n,n-1) = -1; B(n,:) = 0;

    c_raw = zeros(n, m);
    c_raw(1,:) = h * T_oven * dx;
    c_raw(n,:) = h * T_oven * dx;

    C = A \ B;
    d = A \ c_raw;
end
