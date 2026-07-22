#=
2020年高教社杯全国大学生数学建模竞赛 A题 炉温曲线
完整求解代码 (Syslab Julia)
=#

using TyMath, TyPlot

# ==================== 常量定义 ====================
const ZL = 30.5     # 小温区长度 (cm)
const GL = 5.0      # 间隙长度 (cm)
const FL = 25.0     # 炉前/炉后区域长度 (cm)
const TL = FL + 11*ZL + 10*GL + FL  # 总长度 = 435.5 cm
const THICK = 0.015  # 焊接区域厚度 (cm) = 0.15 mm
const DX = 1e-4     # 空间步长 (cm)
const DT = 0.5      # 时间步长 (s)
const TA = 25.0     # 环境温度 (°C)
const TS = 30.0     # 传感器启动温度

# 热力学参数 (实验数据拟合结果)
const XM = [6.677003439269690e-04, 2.999997701878186e+04,
            8.120630727730775e-04, 1.499997442761376e+03,
            9.300295943899003e-04, 1.389867556248244e+03,
            8.343291166559698e-04, 6.898624109710258e+02,
            5.307244068332524e-04, 1.265962465478697e+03]

println("="^70)
println("2020 CUMCM 问题A: 炉温曲线建模与优化")
println("="^70)
println("回焊炉总长度: $(TL) cm | PCB厚度: $(THICK) cm")
println("空间步长: $(DX) cm | 时间步长: $(DT) s")

# ==================== 炉内环境温度 ====================
function oven_temp(x, F)
    L, l, s = ZL, GL, FL
    x1, x2 = 0.0, s
    x3 = x2 + 5*L + 4*l    # zone 5 end = 197.5
    x4 = x3 + l             # gap end = 202.5
    x5 = x4 + L             # zone 6 end = 233
    x6 = x5 + l             # gap end = 238
    x7 = x6 + L             # zone 7 end = 268.5
    x8 = x7 + l             # gap end = 273.5
    x9 = x8 + 2*L + l       # zone 9 end = 339.5
    x10 = x9 + l            # gap end = 344.5

    y = zeros(Float64, length(x))
    for i in eachindex(x)
        xi = x[i]
        if xi <= x2
            y[i] = (F[2]-F[1])/s*(xi-x1) + F[1]
        elseif xi <= x3
            y[i] = F[2]
        elseif xi <= x4
            y[i] = (F[3]-F[2])/l*(xi-x3) + F[2]
        elseif xi <= x5
            y[i] = F[3]
        elseif xi <= x6
            y[i] = (F[4]-F[3])/l*(xi-x5) + F[3]
        elseif xi <= x7
            y[i] = F[4]
        elseif xi <= x8
            y[i] = (F[5]-F[4])/l*(xi-x7) + F[4]
        elseif xi <= x9
            y[i] = F[5]
        elseif xi <= x10
            y[i] = (F[6]-F[5])/l*(xi-x9) + F[5]
        else
            y[i] = F[6]
        end
    end
    return y
end

# ==================== Crank-Nicolson 矩阵构建 ====================
function build_cn(n, r, h, m, T_oven)
    A = zeros(n, n)
    B = zeros(n, n)
    # 内点
    for i in 2:n-1
        A[i,i] = 2*(1+r); A[i,i-1] = -r; A[i,i+1] = -r
        B[i,i] = 2*(1-r); B[i,i-1] = r;  B[i,i+1] = r
    end
    # 边界 (对流)
    A[1,1] = 1+h*DX; A[1,2] = -1
    A[n,n] = 1+h*DX; A[n,n-1] = -1

    c_raw = zeros(n, m)
    for j in 1:m
        c_raw[1,j] = h * T_oven[j] * DX
        c_raw[n,j] = h * T_oven[j] * DX
    end
    c = A \ c_raw
    C = A \ B
    return C, c
end

# ==================== 热传导求解器 ====================
function solve_heat(F, v_cms)
    L1 = FL + 5*ZL + 5*GL    # 197.5
    L2 = L1 + ZL + GL        # 233
    L3 = L2 + ZL + GL        # 268.5
    t1, t2, t3 = L1/v_cms, L2/v_cms, L3/v_cms
    total_time = TL / v_cms

    m1 = floor(Int, t1/DT) + 1
    m2 = floor(Int, t2/DT) + 1
    m3 = floor(Int, t3/DT) + 1

    n = ceil(Int, THICK/DX) + 1
    m = floor(Int, total_time/DT) + 1
    kc = ceil(Int, THICK/2/DX)

    # 炉内温度
    tp = v_cms .* (0:total_time/DT) .* DT
    T_oven = oven_temp(tp, F)

    # 温度场
    u = fill(TA, n, m)
    T_center = fill(TA, m)

    # 5个区域的热参数
    a2 = [XM[1], XM[3], XM[5], XM[7], XM[9]]
    hv = [XM[2], XM[4], XM[6], XM[8], XM[10]]
    rv = a2 .* DT ./ (DX^2)

    C1, c1 = build_cn(n, rv[1], hv[1], m, T_oven)
    C2, c2 = build_cn(n, rv[2], hv[2], m, T_oven)
    C3, c3 = build_cn(n, rv[3], hv[3], m, T_oven)
    C4, c4 = build_cn(n, rv[4], hv[4], m, T_oven)
    C5, c5 = build_cn(n, rv[5], hv[5], m, T_oven)

    for j in 1:m1-1
        u[:,j+1] = C1 * u[:,j] + c1[:,j+1]
        T_center[j+1] = u[kc, j+1]
    end
    for j in m1:m2-1
        u[:,j+1] = C2 * u[:,j] + c2[:,j+1]
        T_center[j+1] = u[kc, j+1]
    end
    for j in m2:m3-1
        u[:,j+1] = C3 * u[:,j] + c3[:,j+1]
        T_center[j+1] = u[kc, j+1]
    end
    for j in m3:m-1
        if T_center[j] >= T_center[j-1]
            u[:,j+1] = C4 * u[:,j] + c4[:,j+1]
        else
            u[:,j+1] = C5 * u[:,j] + c5[:,j+1]
        end
        T_center[j+1] = u[kc, j+1]
    end

    return T_center, T_oven
end

# ==================== 分析函数 ====================
function analyze(T)
    T = T[T .>= TS]
    Tmax, imax = findmax(T)
    slopes = abs.(T[2:end] .- T[1:end-1]) ./ DT
    max_slope = maximum(slopes)

    rising = vcat(false, T[2:end] .>= T[1:end-1])
    in_range = (T .>= 150) .& (T .<= 190)
    t_150_190 = sum(in_range .& rising) * DT

    above = T .>= 217
    t_above = sum(above) * DT

    above_vals = T[above] .- 217
    if isempty(above_vals)
        area = Inf
    else
        peak_idx = argmax(above_vals)
        if peak_idx > 1
            area = (sum(above_vals[1:peak_idx]) - (above_vals[1]+above_vals[peak_idx])/2) * DT
        else
            area = above_vals[1] * DT / 2
        end
    end

    return (Tmax=Tmax, max_slope=max_slope, t_150_190=t_150_190,
            t_above=t_above, area=area, peak_idx=imax)
end

function check_constraints(m)
    (240 <= m.Tmax <= 250) &&
    (40 <= m.t_above <= 90) &&
    (m.max_slope <= 3.0) &&
    (60 <= m.t_150_190 <= 120)
end

# ==================== 问题1 ====================
println("\n" * "="^70)
println("问题1: 给定参数的炉温曲线")
println("="^70)

v1 = 78.0 / 60.0
F1 = [TA, 173.0, 198.0, 230.0, 257.0, TA]
println("v=78 cm/min, T=[173,173,173,173,173,198,230,257,257,25,25]")

T1, oven1 = solve_heat(F1, v1)
m1 = analyze(T1)

# 关键位置和温度
pos = zeros(4)
pos[1] = FL + 2*(ZL+GL) + ZL/2
pos[2] = FL + 5*ZL + 4*GL + GL + ZL/2
pos[3] = FL + 5*ZL + 4*GL + GL + ZL + GL + ZL/2
pos[4] = FL + 5*ZL + 4*GL + GL + ZL + GL + ZL + GL + ZL

t_pos = pos ./ v1
t_raw = collect(0:DT:TL/v1)

function interp_t(t_target, t_vec, T_vec)
    idx = findlast(t -> t <= t_target, t_vec)
    if idx === nothing
        return T_vec[1]
    elseif idx == length(t_vec)
        return T_vec[end]
    end
    alpha = (t_target - t_vec[idx]) / (t_vec[idx+1] - t_vec[idx])
    return T_vec[idx] + alpha * (T_vec[idx+1] - T_vec[idx])
end

pos_names = ["小温区3中点", "小温区6中点", "小温区7中点", "小温区8结束处"]
for i in 1:4
    T_val = interp_t(t_pos[i], t_raw, T1)
    println("  $(pos_names[i]): x=$(round(pos[i],digits=1))cm, t=$(round(t_pos[i],digits=1))s, T=$(round(T_val,digits=1))°C")
end

println("\n炉温曲线指标:")
println("  峰值温度: $(round(m1.Tmax,digits=1))°C")
println("  最大斜率: $(round(m1.max_slope,digits=3))°C/s")
println("  150-190°C(升温): $(round(m1.t_150_190,digits=1))s")
println("  >217°C时间: $(round(m1.t_above,digits=1))s")
println("  >217°C到峰值面积: $(round(m1.area,digits=1))°C·s")

# 生成result.csv
idx_start = findfirst(t -> t >= 30.0, T1)
if idx_start !== nothing
    open("result.csv", "w") do io
        write(io, "时间(s),温度(°C)\n")
        for i in idx_start:length(T1)
            write(io, "$(round(t_raw[i],digits=1)),$(round(T1[i],digits=4))\n")
        end
    end
    println("  result.csv 已生成，$(length(T1)-idx_start+1)行")
end

# 问题1图
figure()
subplot(1,2,1)
plot(t_raw, T1, "r-", linewidth=1.5)
hold("on")
plot(t_raw, oven1, "b--", linewidth=1)
yline(217, "g--", linewidth=0.8)
yline(30, ":", linewidth=0.5)
for i in 1:4
    plot([t_pos[i]], [interp_t(t_pos[i], t_raw, T1)], "o", markersize=8)
end
hold("off")
xlabel("时间 (s)"); ylabel("温度 (°C)")
title("问题1: 炉温曲线 (时间-温度)")
legend("焊接区域中心温度", "炉内环境温度", "T=217°C", "T=30°C")
grid(true)

subplot(1,2,2)
x_pos = v1 .* t_raw
plot(x_pos, T1, "r-", linewidth=1.5)
hold("on")
plot(x_pos, oven1, "b--", linewidth=1)
yline(217, "g--", linewidth=0.8)
hold("off")
xlabel("位置 (cm)"); ylabel("温度 (°C)")
title("位置-温度曲线")
legend("焊接区域中心温度", "炉内环境温度")
grid(true)
print("q1_furnace_curve", "-dpng")
println("  图已保存: q1_furnace_curve.png")

# ==================== 问题2 ====================
println("\n" * "="^70)
println("问题2: 最大传送带速度")
println("="^70)

F2 = [TA, 182.0, 203.0, 237.0, 254.0, TA]
println("T=[182,182,182,182,182,203,237,254,254,25,25]")

function is_feasible(v_cm_min, F)
    T_out, _ = solve_heat(F, v_cm_min/60.0)
    return check_constraints(analyze(T_out))
end

# 扫描速度范围
println("\n速度扫描 (65-100 cm/min):")
println("  速度 | 峰值°C | 斜率°C/s | 150-190s | >217s | 可行")
for v_cm in 65:5:100
    Ttmp, _ = solve_heat(F2, v_cm/60.0)
    mt = analyze(Ttmp)
    ok = check_constraints(mt)
    println("  $(v_cm)  | $(round(mt.Tmax,digits=1))  | $(round(mt.max_slope,digits=3))   | $(round(mt.t_150_190,digits=1))    | $(round(mt.t_above,digits=1))  | $ok")
end

# 二分搜索最大速度
v_lo, v_hi = 65.0, 100.0
if is_feasible(v_hi, F2)
    v_max = v_hi
else
    while v_hi - v_lo > 1e-4
        v_mid = (v_lo + v_hi) / 2
        if is_feasible(v_mid, F2)
            v_lo = v_mid
        else
            v_hi = v_mid
        end
    end
    v_max = v_lo
end

println("\n最大允许速度: $(round(v_max,digits=3)) cm/min")

# 验证
T2, oven2 = solve_heat(F2, v_max/60.0)
m2 = analyze(T2)
println("验证指标: Tmax=$(round(m2.Tmax,digits=1))°C, " *
        "斜率=$(round(m2.max_slope,digits=3))°C/s, " *
        "t150-190=$(round(m2.t_150_190,digits=1))s, " *
        "t>217=$(round(m2.t_above,digits=1))s")

# 问题2图
figure()
v_list = collect(65:5:100)
scan_data = [begin
    Ttmp, _ = solve_heat(F2, v/60.0)
    analyze(Ttmp)
end for v in v_list]

subplot(2,2,1)
plot(v_list, [s.max_slope for s in scan_data], "b-o", markersize=5)
yline(3, "r--", linewidth=1)
xlabel("速度(cm/min)"); ylabel("斜率(°C/s)"); title("最大斜率")
legend(["斜率", "上限"]); grid(true)

subplot(2,2,2)
plot(v_list, [s.Tmax for s in scan_data], "r-o", markersize=5)
yline(250, "r--"); yline(240, "--", color=[1,0.5,0])
xlabel("速度(cm/min)"); ylabel("温度(°C)"); title("峰值温度"); grid(true)

subplot(2,2,3)
plot(v_list, [s.t_above for s in scan_data], "g-o", markersize=5)
yline(90, "r--"); yline(40, "--", color=[1,0.5,0])
xlabel("速度(cm/min)"); ylabel("时间(s)"); title(">217°C时间"); grid(true)

subplot(2,2,4)
plot(v_list, [s.t_150_190 for s in scan_data], "m-o", markersize=5)
yline(120, "r--"); yline(60, "--", color=[1,0.5,0])
xlabel("速度(cm/min)"); ylabel("时间(s)"); title("150-190°C时间"); grid(true)
print("q2_speed_analysis", "-dpng")
println("  图已保存: q2_speed_analysis.png")

# ==================== 问题3 ====================
println("\n" * "="^70)
println("问题3: 最小化>217°C到峰值的面积")
println("="^70)

function objective_q3(x)
    T15, T6, T7, T89, vcm = x
    F = [TA, T15, T6, T7, T89, TA]
    T_out, _ = solve_heat(F, vcm/60.0)
    m = analyze(T_out)
    check_constraints(m) ? m.area : Inf
end

lb = [165.0, 185.0, 225.0, 245.0, 65.0]
ub = [185.0, 205.0, 245.0, 265.0, 100.0]

println("粗网格搜索...")
best_area, best_x = Inf, zeros(5)
Tg = [range(lb[i], ub[i], length=5) for i in 1:4]
vg = range(65, 100, length=7)

cnt = 0
for T15 in Tg[1], T6 in Tg[2], T7 in Tg[3], T89 in Tg[4], vcm in vg
    cnt += 1
    area = objective_q3([T15, T6, T7, T89, vcm])
    if area < best_area
        best_area = area
        best_x = [T15, T6, T7, T89, vcm]
    end
end
println("粗搜索: $(cnt)点, 最优面积=$(round(best_area,digits=1))")

# 精细搜索
println("精细搜索...")
for T15 in range(max(lb[1],best_x[1]-2), min(ub[1],best_x[1]+2), length=7)
    for T6 in range(max(lb[2],best_x[2]-2), min(ub[2],best_x[2]+2), length=7)
        for T7 in range(max(lb[3],best_x[3]-2), min(ub[3],best_x[3]+2), length=7)
            for T89 in range(max(lb[4],best_x[4]-2), min(ub[4],best_x[4]+2), length=7)
                for vcm in range(max(65.0,best_x[5]-3), min(100.0,best_x[5]+3), length=7)
                    area = objective_q3([T15, T6, T7, T89, vcm])
                    if area < best_area
                        best_area = area
                        best_x = [T15, T6, T7, T89, vcm]
                    end
                end
            end
        end
    end
end

println("\n问题3 最优解:")
println("  温区1-5: $(round(best_x[1],digits=1))°C")
println("  温区6:   $(round(best_x[2],digits=1))°C")
println("  温区7:   $(round(best_x[3],digits=1))°C")
println("  温区8-9: $(round(best_x[4],digits=1))°C")
println("  速度:    $(round(best_x[5],digits=2)) cm/min")
println("  面积:    $(round(best_area,digits=1)) °C·s")

F3 = [TA, best_x[1], best_x[2], best_x[3], best_x[4], TA]
T3, oven3 = solve_heat(F3, best_x[5]/60.0)
m3 = analyze(T3)
println("  峰值温度: $(round(m3.Tmax,digits=1))°C")
println("  >217°C时间: $(round(m3.t_above,digits=1))s")

# 问题3图
figure()
t3 = collect(0:DT:TL/(best_x[5]/60.0))

subplot(1,2,1)
plot(t3, T3, "r-", linewidth=1.5)
hold("on")
plot(t3, oven3, "b--", linewidth=1)
yline(217, "g--", linewidth=0.8)

# 填充面积
idx_s3 = findfirst(t -> T3[t] >= TS, 1:length(T3))
idx_s3 = idx_s3 === nothing ? 1 : idx_s3
peak_idx3 = argmax(T3[idx_s3:end]) + idx_s3 - 1
above_fill = copy(T3)
above_fill[T3 .< 217] .= 217
x_fill = t3[1:peak_idx3]
y_fill_top = above_fill[1:peak_idx3]
y_fill_bottom = 217 * ones(peak_idx3)
# Mark peak
plot([t3[peak_idx3]], [T3[peak_idx3]], "ro", markersize=8)
hold("off")
xlabel("时间(s)"); ylabel("温度(°C)")
title("问题3: 最优炉温曲线")
legend(["焊接区域中心温度", "炉内环境温度", "T=217°C"])
grid(true)

subplot(1,2,2)
x3 = (best_x[5]/60.0) .* t3
plot(x3, T3, "r-", linewidth=1.5)
hold("on")
plot(x3, oven3, "b--", linewidth=1)
yline(217, "g--", linewidth=0.8)
hold("off")
xlabel("位置(cm)"); ylabel("温度(°C)")
title("位置-温度曲线")
legend(["焊接区域中心温度", "炉内环境温度"])
grid(true)
print("q3_optimal", "-dpng")
println("  图已保存: q3_optimal.png")

# ==================== 问题4 ====================
println("\n" * "="^70)
println("问题4: 带对称性约束的优化")
println("="^70)

function compute_symmetry(T)
    T = T[T .>= TS]
    above = T[T .>= 217] .- 217
    if isempty(above)
        return 1.0, Inf
    end

    pk = argmax(above)
    n = length(above)

    AL = pk > 1 ? (sum(above[1:pk]) - (above[1]+above[pk])/2)*DT : above[1]*DT/2
    AR = n-pk > 1 ? (sum(above[pk:n]) - (above[pk]+above[n])/2)*DT : above[pk]*DT/2

    sigma1 = abs(AL-AR) / max(abs(AL), abs(AR))
    sigma2 = abs((n-pk) - (pk-1)) / max(pk-1, n-pk)
    sigma = max(sigma1, sigma2)
    return sigma, AL+AR
end

function objective_q4(x, sigma_max)
    T15, T6, T7, T89, vcm = x
    F = [TA, T15, T6, T7, T89, TA]
    T_out, _ = solve_heat(F, vcm/60.0)
    m = analyze(T_out)
    if !check_constraints(m)
        return Inf
    end

    sigma, _ = compute_symmetry(T_out)
    if sigma > sigma_max
        return Inf
    end

    return m.area
end

println("\n基准解 (来自问题3):")
sigma_base, _ = compute_symmetry(T3)
println("  对称性σ: $(round(sigma_base,digits=4)), 面积: $(round(best_area,digits=1))")

# 逐步收紧对称性
best_q4 = nothing
for sig_target in [0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    best_a4 = Inf
    best_x4 = zeros(5)

    Tg = [range(lb[i], ub[i], length=6) for i in 1:4]
    vg = range(65, 100, length=8)

    for T15 in Tg[1], T6 in Tg[2], T7 in Tg[3], T89 in Tg[4], vcm in vg
        a = objective_q4([T15, T6, T7, T89, vcm], sig_target)
        if a < best_a4
            best_a4 = a
            best_x4 = [T15, T6, T7, T89, vcm]
        end
    end

    if best_a4 < Inf
        Ftmp = [TA, best_x4[1], best_x4[2], best_x4[3], best_x4[4], TA]
        Ttmp, _ = solve_heat(Ftmp, best_x4[5]/60.0)
        sig_actual, _ = compute_symmetry(Ttmp)
        best_q4 = (sig_target, best_x4, best_a4, sig_actual)
        println("  σ≤$sig_target: 面积=$(round(best_a4,digits=1)), " *
                "实际σ=$(round(sig_actual,digits=4)), " *
                "T15=$(round(best_x4[1],digits=1)), T6=$(round(best_x4[2],digits=1)), " *
                "T7=$(round(best_x4[3],digits=1)), T89=$(round(best_x4[4],digits=1)), " *
                "v=$(round(best_x4[5],digits=2))")
    else
        println("  σ≤$sig_target: 无可行解")
    end
end

if best_q4 !== nothing
    _, bx4, ba4, sig4 = best_q4
    println("\n问题4 最优解:")
    println("  温区1-5: $(round(bx4[1],digits=1))°C")
    println("  温区6:   $(round(bx4[2],digits=1))°C")
    println("  温区7:   $(round(bx4[3],digits=1))°C")
    println("  温区8-9: $(round(bx4[4],digits=1))°C")
    println("  速度:    $(round(bx4[5],digits=2)) cm/min")
    println("  面积:    $(round(ba4,digits=1)) °C·s")
    println("  对称性σ: $(round(sig4,digits=4))")

    F4 = [TA, bx4[1], bx4[2], bx4[3], bx4[4], TA]
    T4, oven4 = solve_heat(F4, bx4[5]/60.0)
    m4 = analyze(T4)

    # 问题4图
    figure()
    t4 = collect(0:DT:TL/(bx4[5]/60.0))

    subplot(1,2,1)
    plot(t4, T4, "r-", linewidth=1.5)
    hold("on")
    plot(t4, oven4, "b--", linewidth=1)
    yline(217, "g--", linewidth=0.8)

    pk4 = argmax(T4)
    idx_s4 = findfirst(t -> T4[t] >= TS, 1:length(T4))
    idx_s4 = idx_s4 === nothing ? 1 : idx_s4
    above4 = copy(T4)
    above4[T4 .< 217] .= 217

    # 标记峰值
    plot([t4[pk4]], [T4[pk4]], "ro", markersize=8)
    hold("off")
    xlabel("时间(s)"); ylabel("温度(°C)")
    title("问题4: 对称最优炉温曲线")
    legend(["焊接区域中心温度", "炉内环境温度", "T=217°C", "峰值"])
    grid(true)

    subplot(1,2,2)
    T_s = T4[idx_s4:end]
    t_s = t4[idx_s4:end]
    pk = argmax(T_s)
    t_left = t_s[1:pk] .- t_s[pk]
    t_right = t_s[pk:end] .- t_s[pk]
    plot(t_left, T_s[1:pk], "r-", linewidth=1.5)
    hold("on")
    plot(-t_right, T_s[pk:end], "b--", linewidth=1.5)
    yline(217, "g--", linewidth=0.8)
    hold("off")
    xlabel("相对峰值时间(s)"); ylabel("温度(°C)")
    title("对称性分析 (σ=$(round(sig4,digits=4)))")
    legend(["峰值左侧", "峰值右侧(镜像)"])
    grid(true)
    print("q4_symmetric_optimal", "-dpng")
    println("  图已保存: q4_symmetric_optimal.png")

    sigma3, _ = compute_symmetry(T3)
    println("\n问题3 vs 问题4:")
    println("  问题3: 面积=$(round(best_area,digits=1)), σ=$(round(sigma3,digits=4))")
    println("  问题4: 面积=$(round(ba4,digits=1)), σ=$(round(sig4,digits=4))")
end

println("\n" * "="^70)
println("全部问题求解完成!")
println("="^70)
