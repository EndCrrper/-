#=
炉温曲线建模 - 核心求解模块
基于一维热传导方程和Crank-Nicolson有限差分法
=#

using TyMath, TyPlot, TyOptimization

# ==================== 常量定义 ====================
const zone_length = 30.5   # 每个小温区长度 (cm)
const gap_length = 5.0     # 小温区间隙 (cm)
const front_length = 25.0  # 炉前区域长度 (cm)
const rear_length = 25.0   # 炉后区域长度 (cm)
const total_length = front_length + 11*zone_length + 10*gap_length + rear_length  # 总长度 = 435.5 cm
const thickness = 0.015    # 焊接区域厚度 (cm) = 0.15 mm
const dx = 1e-4            # 空间步长 (cm)
const dt = 0.5             # 时间步长 (s)
const T_ambient = 25.0     # 环境/车间温度 (°C)
const T_sensor_start = 30.0  # 传感器开始工作温度

# 各温区分界点位置 (cm)
const x_boundaries = let
    x = zeros(12)
    x[1] = 0.0
    x[2] = front_length  # 25
    for i in 1:5
        x[2+i] = x[1+i] + zone_length
        if i < 5
            x[2+i] += gap_length
        end
    end
    # After zone 5: +gap, then zone 6
    x[8] = x[7] + gap_length + zone_length
    # After zone 6: +gap, then zone 7
    x[9] = x[8] + gap_length + zone_length
    # After zone 7: +gap, then zones 8-9
    x[10] = x[9] + gap_length + 2*zone_length + gap_length
    # After zone 9: +gap, then zones 10-11
    x[11] = x[10] + gap_length + 2*zone_length + gap_length
    x[12] = x[11] + rear_length
    x
end

# ==================== 炉内环境温度分布函数 ====================
"""
计算炉内位置x处的环境温度
F: [T_front, T_zone1_5, T_zone6, T_zone7, T_zone8_9, T_rear]
   其中 T_front = T_rear = 25°C
"""
function oven_temperature(x, F)
    l = gap_length
    L = zone_length
    s = front_length

    x1 = 0.0
    x2 = s                                    # 25 - 炉前结束
    x3 = x2 + 5*L + 4*l                       # 197.5 - 温区5结束
    x4 = x3 + l                                # 202.5 - 间隙 (到温区6)
    x5 = x4 + L                                # 233 - 温区6结束
    x6 = x5 + l                                # 238 - 间隙 (到温区7)
    x7 = x6 + L                                # 268.5 - 温区7结束
    x8 = x7 + l                                # 273.5 - 间隙 (到温区8)
    x9 = x8 + 2*L + l                          # 339.5 - 温区9结束
    x10 = x9 + l                               # 344.5 - 间隙 (到温区10)

    y = zeros(length(x))
    for i in eachindex(x)
        xi = x[i]
        if xi <= x2
            y[i] = (F[2] - F[1]) / s * (xi - x1) + F[1]
        elseif xi <= x3
            y[i] = F[2]
        elseif xi <= x4
            y[i] = (F[3] - F[2]) / l * (xi - x3) + F[2]
        elseif xi <= x5
            y[i] = F[3]
        elseif xi <= x6
            y[i] = (F[4] - F[3]) / l * (xi - x5) + F[3]
        elseif xi <= x7
            y[i] = F[4]
        elseif xi <= x8
            y[i] = (F[5] - F[4]) / l * (xi - x7) + F[4]
        elseif xi <= x9
            y[i] = F[5]
        elseif xi <= x10
            y[i] = (F[6] - F[5]) / l * (xi - x9) + F[5]
        else
            y[i] = F[6]
        end
    end
    return y
end

# ==================== Crank-Nicolson 热传导求解器 ====================
"""
使用Crank-Nicolson有限差分法求解一维热传导方程

参数:
- a2_vals: 5个热扩散系数 a² = k/(ρc) 的值
- h_vals: 5个热交换系数 h = h₁/k 的值
- F: 各温区设定温度 [T_front, T1_5, T6, T7, T8_9, T_rear]
- v: 传送带速度 (cm/s)
- total_time: 总仿真时间 (s)

返回:
- t_out: 从传感器启动开始的时间序列
- T_center: 焊接区域中心温度序列
"""
function solve_heat_equation(a2_vals, h_vals, F, v, total_time)
    # 各段的分界时间
    # 段1: 0 到 温区5结束 (x=197.5)
    L1 = 25 + 5*30.5 + 5*5    # = 197.5 cm
    L2 = L1 + 30.5 + 5         # = 233 cm (zone 6 end)
    L3 = L2 + 30.5 + 5         # = 268.5 cm (zone 7 end)

    t1 = L1 / v
    t2 = L2 / v
    t3 = L3 / v

    m1 = floor(Int, t1/dt) + 1
    m2 = floor(Int, t2/dt) + 1
    m3 = floor(Int, t3/dt) + 1

    n = ceil(Int, thickness/dx) + 1  # 空间网格数
    m = floor(Int, total_time/dt) + 1  # 时间步数

    k_center = ceil(Int, thickness/2/dx)  # 中心点索引

    # 炉内环境温度序列
    time_points = v .* (0:total_time/dt) .* dt  # 位置 = 速度 × 时间
    T_oven = oven_temperature(time_points, F)

    # 温度场初始化
    u = zeros(n, m)
    u[:, 1] .= T_ambient
    t_out = fill(T_ambient, m)

    # 预处理各段的Crank-Nicolson矩阵
    r_vals = a2_vals .* dt / (dx^2)

    # 段1: 前区 + 温区1-5
    h = h_vals[1]; r = r_vals[1]
    A1, B1, c1 = build_cn_matrices(n, r, h, m, T_oven, dx)

    # 段2: 温区6
    h = h_vals[2]; r = r_vals[2]
    A2, B2, c2 = build_cn_matrices(n, r, h, m, T_oven, dx)

    # 段3: 温区7
    h = h_vals[3]; r = r_vals[3]
    A3, B3, c3 = build_cn_matrices(n, r, h, m, T_oven, dx)

    # 段4: 温区8-9 (升温段)
    h = h_vals[4]; r = r_vals[4]
    A4, B4, c4 = build_cn_matrices(n, r, h, m, T_oven, dx)

    # 段5: 温区10-11 (降温段)
    h = h_vals[5]; r = r_vals[5]
    A5, B5, c5 = build_cn_matrices(n, r, h, m, T_oven, dx)

    # 时间步进 - 段1
    C1 = A1 \ B1
    for j in 1:m1-1
        u[:, j+1] = C1 * u[:, j] + c1[:, j+1]
        t_out[j+1] = u[k_center, j+1]
    end

    # 时间步进 - 段2
    C2 = A2 \ B2
    for j in m1:m2-1
        u[:, j+1] = C2 * u[:, j] + c2[:, j+1]
        t_out[j+1] = u[k_center, j+1]
    end

    # 时间步进 - 段3
    C3 = A3 \ B3
    for j in m2:m3-1
        u[:, j+1] = C3 * u[:, j] + c3[:, j+1]
        t_out[j+1] = u[k_center, j+1]
    end

    # 时间步进 - 段4和段5 (根据温度变化趋势切换)
    C4 = A4 \ B4
    C5 = A5 \ B5
    for j in m3:m-1
        if t_out[j] >= t_out[j-1]
            u[:, j+1] = C4 * u[:, j] + c4[:, j+1]
        else
            u[:, j+1] = C5 * u[:, j] + c5[:, j+1]
        end
        t_out[j+1] = u[k_center, j+1]
    end

    return t_out, T_oven
end

"""
构建Crank-Nicolson差分矩阵

热传导方程: ∂T/∂t = a² ∂²T/∂z²
边界条件 (对流): -k ∂T/∂z = h₁(T_oven - T)  at z=0
                  k ∂T/∂z = h₁(T_oven - T)  at z=l

Crank-Nicolson格式:
(I + r*L/2) * u^{n+1} = (I - r*L/2) * u^n + b
其中 r = a²*dt/(dx²)

边界离散:
z=0: (-3u₀ + 4u₁ - u₂)/(2dx) = h*(T_oven - u₀) → u₀ 的修正方程
z=l: (3u_n - 4u_{n-1} + u_{n-2})/(2dx) = h*(T_oven - u_n)

简化为:
z=0: -(1+h*dx)u₀ + u₁ = -h*dx*T_oven
z=l: -u_{n-1} + (1+h*dx)u_n = h*dx*T_oven
"""
function build_cn_matrices(n, r, h, m, T_oven, dx)
    # 构建三对角矩阵 A = I + r*L/2 (隐式部分)
    A = zeros(n, n)
    B = zeros(n, n)

    # 内点
    for i in 2:n-1
        A[i, i] = 2*(1 + r)
        A[i, i-1] = -r
        A[i, i+1] = -r

        B[i, i] = 2*(1 - r)
        B[i, i-1] = r
        B[i, i+1] = r
    end

    # 边界点 (对流边界条件)
    # z=0: 使用三点前向差分 -3T₀+4T₁-T₂ = 2dx*h*(T_oven-T₀)
    # 整理: (1+h*dx)*T₀ - T₁ = h*dx*T_oven  (在C-N格式中等效)
    A[1, 1] = 1 + h*dx
    A[1, 2] = -1
    B[1, 1] = 0
    B[1, 2] = 0

    # z=l: 使用三点后向差分 3Tₙ-4Tₙ₋₁+Tₙ₋₂ = -2dx*h*(T_oven-Tₙ)
    A[n, n] = 1 + h*dx
    A[n, n-1] = -1
    B[n, n] = 0
    B[n, n-1] = 0

    # 边界源项
    c = zeros(n, m)
    for j in 1:m
        c[1, j] = h * T_oven[j] * dx
        c[n, j] = h * T_oven[j] * dx
    end
    c = A \ c  # 预求解源项

    return A, B, c
end

# ==================== 工具函数 ====================

"""
根据传感器条件截取温度序列 (从 T >= 30°C 开始)
"""
function extract_sensor_data(t_out, start_temp=T_sensor_start)
    idx = findfirst(t -> t >= start_temp, t_out)
    if idx === nothing
        return t_out, 1
    end
    return t_out[idx:end], idx
end

"""
分析炉温曲线的各项指标
"""
function analyze_curve(T)
    T = T[T .>= T_sensor_start]

    # 峰值温度
    Tmax, imax = findmax(T)

    # 最大升降温斜率
    slopes = abs.(T[2:end] .- T[1:end-1]) ./ dt
    max_slope = maximum(slopes)

    # 上升段中150-190°C的时间
    rising = T[2:end] .>= T[1:end-1]
    rising_150_190 = T[rising] .>= 150 .&& T[rising] .<= 190
    # 更准确的计算
    in_range = (T .>= 150) .& (T .<= 190)
    rising_idx = vcat(false, T[2:end] .>= T[1:end-1])
    t_150_190 = sum(in_range .& rising_idx) * dt

    # 温度大于217°C的时间
    t_above_217 = sum(T .>= 217) * dt

    # 超过217°C到峰值温度的面积
    above_217 = T .- 217
    above_217 = above_217[above_217 .> 0]
    peak_idx = findfirst(t -> t == maximum(above_217), above_217)
    if peak_idx !== nothing && peak_idx > 1
        area = (sum(above_217[1:peak_idx]) - (above_217[1] + above_217[peak_idx])/2) * dt
    elseif peak_idx !== nothing
        area = (above_217[1] + above_217[peak_idx]) * dt / 2
    else
        area = 0.0
    end

    return (Tmax=Tmax, max_slope=max_slope, t_150_190=t_150_190,
            t_above_217=t_above_217, area=area, peak_idx=imax)
end

"""
检查是否满足制程界限
"""
function check_constraints(metrics)
    (240 <= metrics.Tmax <= 250) &&
    (40 <= metrics.t_above_217 <= 90) &&
    (metrics.max_slope <= 3.0) &&
    (60 <= metrics.t_150_190 <= 120)
end

println("核心求解模块加载完成。")
println("回焊炉总长度: $(total_length) cm")
println("PCB厚度: $(thickness) cm")
