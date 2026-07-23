function [T_center, T_oven, t_full] = solve_heat(F, v_cms, xm)
% solve_heat - 使用 Crank-Nicolson 有限差分法求解一维热传导方程
%
% 输入:
%   F     : 各段设定温度 [T_front, T1_5, T6, T7, T8_9, T_rear]
%   v_cms : 传送带速度 (cm/s)
%   xm    : 10个热力学参数 [a1,h1, a2,h2, a3,h3, a4,h4, a5,h5]
%           a_i : 热扩散系数 (cm/s^(1/2)), 在CN格式中平方使用 a_i^2
%           h_i : 无量纲换热系数 (1/cm)
%
% 输出:
%   T_center: 焊接区域中心完整温度序列 (°C), 从 t=0 开始
%   T_oven  : 炉内环境温度序列 (°C)
%   t_full  : 时间序列 (s), 从 t=0 开始

    % ==================== 常量定义 ====================
    L_total = 25 + 11*30.5 + 10*5 + 25;  % 回焊炉总长度 = 435.5 cm
    thickness = 0.015;    % 焊接区域厚度 (cm)
    dx = 1e-4;            % 空间步长 (cm)
    dt = 0.5;             % 时间步长 (s)
    T_ambient = 25;       % 车间环境温度 (°C)
    T_sensor_start = 30;  % 传感器启动温度 (°C)

    % ==================== 总时间和网格 ====================
    total_time = L_total / v_cms;  % 总过炉时间
    n = ceil(thickness / dx) + 1;  % 空间网格点数 (约151)
    m = floor(total_time / dt) + 1;% 时间步数
    k_center = ceil(thickness / 2 / dx); % 中心点网格索引 (约75)

    % ==================== 炉内环境温度序列 ====================
    t_vec = (0:m-1) * dt;                  % 时间向量
    x_vec = v_cms * t_vec;                 % 位置向量 x = v*t
    T_oven = oven_temp(x_vec, F);          % 炉内环境温度

    % ==================== 各段分界时间 ====================
    L1 = 25 + 5*30.5 + 5*5;    % = 197.5 cm (温区5结束)
    L2 = L1 + 30.5 + 5;         % = 233 cm   (温区6结束)
    L3 = L2 + 30.5 + 5;         % = 268.5 cm (温区7结束)

    t1 = L1 / v_cms;  t2 = L2 / v_cms;  t3 = L3 / v_cms;
    m1 = floor(t1/dt) + 1;  % 段1结束时间步
    m2 = floor(t2/dt) + 1;  % 段2结束时间步
    m3 = floor(t3/dt) + 1;  % 段3结束时间步

    % ==================== 提取热力学参数 ====================
    % xm = [a1, h1, a2, h2, a3, h3, a4, h4, a5, h5]
    % 注意: a 需要平方后使用: r = a^2 * dt / dx^2
    a_vals = [xm(1), xm(3), xm(5), xm(7), xm(9)];
    h_vals = [xm(2), xm(4), xm(6), xm(8), xm(10)];
    r_vals = a_vals.^2 * dt / (dx^2);  % CN格式的无量纲参数

    % ==================== 温度场初始化 ====================
    u = zeros(n, m);           % 完整温度场
    u(:, 1) = T_ambient;       % 初始条件: 全部为车间温度
    T_center = ones(m, 1) * T_ambient;  % 中心温度序列

    % ==================== 构建各段 CN 矩阵并求解 ====================

    % ----- 段1: 炉前 + 温区1~5 (x: 0 ~ 197.5 cm) -----
    [C1, d1] = build_cn_system(n, r_vals(1), h_vals(1), m, T_oven, dx);
    for j = 1:m1-1
        u(:, j+1) = C1 * u(:, j) + d1(:, j+1);
        T_center(j+1) = u(k_center, j+1);
    end

    % ----- 段2: 温区6 (x: 197.5 ~ 233 cm) -----
    [C2, d2] = build_cn_system(n, r_vals(2), h_vals(2), m, T_oven, dx);
    for j = m1:m2-1
        u(:, j+1) = C2 * u(:, j) + d2(:, j+1);
        T_center(j+1) = u(k_center, j+1);
    end

    % ----- 段3: 温区7 (x: 233 ~ 268.5 cm) -----
    [C3, d3] = build_cn_system(n, r_vals(3), h_vals(3), m, T_oven, dx);
    for j = m2:m3-1
        u(:, j+1) = C3 * u(:, j) + d3(:, j+1);
        T_center(j+1) = u(k_center, j+1);
    end

    % ----- 段4&5: 温区8~11 (x: 268.5 ~ 435.5 cm) -----
    % 根据温度变化趋势自动切换升温段(段4)和降温段(段5)参数
    [C4, d4] = build_cn_system(n, r_vals(4), h_vals(4), m, T_oven, dx);
    [C5, d5] = build_cn_system(n, r_vals(5), h_vals(5), m, T_oven, dx);
    for j = m3:m-1
        if T_center(j) >= T_center(j-1)
            % 升温阶段: 使用段4参数 (回流区)
            u(:, j+1) = C4 * u(:, j) + d4(:, j+1);
        else
            % 降温阶段: 使用段5参数 (冷却区)
            u(:, j+1) = C5 * u(:, j) + d5(:, j+1);
        end
        T_center(j+1) = u(k_center, j+1);
    end

    % ==================== 截取传感器工作后的数据 ====================
    % 返回完整序列，由调用方决定是否截取 T>=30°C

    % 输出时间序列
    if nargout >= 3
        t_full = (0:m-1) * dt;
    end
end

% ==================== 子函数: 构建 Crank-Nicolson 系统矩阵 ====================
function [C, d] = build_cn_system(n, r, h, m, T_oven, dx)
    % 构建 A*T^{j+1} = B*T^j + c^{j+1} 的三对角系统
    % 返回预分解的 C = A\B 和 d = A\c

    % ---- 构建矩阵 A (隐式部分) 和 B (显式部分) ----
    A = zeros(n, n);
    B = zeros(n, n);

    % 内点 (i = 2, 3, ..., n-1)
    for i = 2:n-1
        A(i, i)   = 2 * (1 + r);
        A(i, i-1) = -r;
        A(i, i+1) = -r;

        B(i, i)   = 2 * (1 - r);
        B(i, i-1) = r;
        B(i, i+1) = r;
    end

    % 边界点: 对流换热边界条件 (一阶离散)
    % z=0: (1 + h*dx)*T_1 - T_2 = h*dx*T_oven
    A(1, 1) = 1 + h * dx;
    A(1, 2) = -1;
    B(1, :) = 0;

    % z=l: (1 + h*dx)*T_n - T_{n-1} = h*dx*T_oven
    A(n, n)   = 1 + h * dx;
    A(n, n-1) = -1;
    B(n, :) = 0;

    % ---- 边界源项向量 c ----
    c_raw = zeros(n, m);
    c_raw(1, :) = h * T_oven * dx;
    c_raw(n, :) = h * T_oven * dx;

    % ---- 预分解 ----
    C = A \ B;      % n×n 矩阵
    d = A \ c_raw;  % n×m 矩阵
end
