#=
问题4: 在问题3基础上增加对称性要求
在满足制程界限的前提下，最小化超过217°C面积，同时使峰值两侧曲线尽量对称
对称性指标: σ = max(|A_L - A_R|/max(A_L, A_R), |T_R - 2T_peak + T_L|/max(T_peak - T_L, T_R - T_peak))
其中 A_L, A_R 分别为峰值左右两侧超过217°C的"等效时间"
=#

include("solution_core.jl")

# 热力学参数
const xm = [6.677003439269690e-04; 2.999997701878186e+04;
            8.120630727730775e-04; 1.499997442761376e+03;
            9.300295943899003e-04; 1.389867556248244e+03;
            8.343291166559698e-04; 6.898624109710258e+02;
            5.307244068332524e-04; 1.265962465478697e+03]

a2_vals = [xm[1], xm[3], xm[5], xm[7], xm[9]]
h_vals = [xm[2], xm[4], xm[6], xm[8], xm[10]]

println("\n" * "="^60)
println("问题4: 带对称性约束的炉温曲线优化")
println("="^60)

# 计算对称性指标
function compute_symmetry(T)
    T = T[T .>= T_sensor_start]
    above = T[T .>= 217] .- 217
    if isempty(above)
        return 1.0, 0.0  # 不对称
    end

    peak_idx = argmax(above)
    n = length(above)

    # 左侧面积 (从217°C交点到峰值)
    if peak_idx > 1
        A_L = (sum(above[1:peak_idx]) - (above[1] + above[peak_idx])/2) * dt
    else
        A_L = above[1] * dt / 2
    end

    # 右侧面积 (从峰值到217°C交点)
    if n - peak_idx > 1
        A_R = (sum(above[peak_idx:n]) - (above[peak_idx] + above[n])/2) * dt
    else
        A_R = above[peak_idx] * dt / 2
    end

    # 对称性指标 σ₁: 面积对称性
    sigma1 = abs(A_L - A_R) / max(abs(A_L), abs(A_R))

    # 对称性指标 σ₂: 时间对称性
    # |T_R - 2*T_peak + T_L| / max(T_peak - T_L, T_R - T_peak)
    # T_L = 217+above[1], T_R = 217+above[n], T_peak = 217+above[peak_idx]
    time_left = (peak_idx - 1) * dt
    time_right = (n - peak_idx) * dt
    sigma2 = abs(time_right - time_left) / max(time_left, time_right)

    # 综合对称性指标
    sigma = max(sigma1, sigma2)

    return sigma, A_L + A_R
end

# 目标函数: 加权组合面积和对称性
function objective_symmetric(x, w_area=0.6, w_sym=0.4)
    T1_5, T6, T7, T8_9, v_cm_min = x
    v = v_cm_min / 60.0
    F = [25.0, T1_5, T6, T7, T8_9, 25.0]
    total_time = total_length / v

    t_out, _ = solve_heat_equation(a2_vals, h_vals, F, v, total_time)

    # 检查制程界限
    metrics = analyze_curve(t_out)
    if !check_constraints(metrics)
        return Inf
    end

    # 对称性
    sigma, _ = compute_symmetry(t_out)

    # 面积
    area = metrics.area

    # 加权目标 (面积归一化)
    area_norm = area / 500.0  # 大约归一化到 [0, 1]
    return w_area * area_norm + w_sym * sigma
end

# 纯面积目标 (有对称性约束时使用)
function objective_area_with_sym_constraint(x, sigma_max=0.1)
    T1_5, T6, T7, T8_9, v_cm_min = x
    v = v_cm_min / 60.0
    F = [25.0, T1_5, T6, T7, T8_9, 25.0]
    total_time = total_length / v

    t_out, _ = solve_heat_equation(a2_vals, h_vals, F, v, total_time)

    # 检查制程界限
    metrics = analyze_curve(t_out)
    if !check_constraints(metrics)
        return Inf
    end

    # 对称性约束
    sigma, _ = compute_symmetry(t_out)
    if sigma > sigma_max
        return Inf
    end

    return metrics.area
end

# 决策变量边界
lb = [165.0, 185.0, 225.0, 245.0, 65.0]
ub = [185.0, 205.0, 245.0, 265.0, 100.0]

# 策略: 先做无对称性约束的优化(问题3), 再逐步收紧对称性要求

println("\n阶段1: 无对称性约束优化(基准)")

# 网格搜索
best_area_base = Inf
best_x_base = zeros(5)

T1_5_grid = range(lb[1], ub[1], length=6)
T6_grid = range(lb[2], ub[2], length=6)
T7_grid = range(lb[3], ub[3], length=6)
T8_9_grid = range(lb[4], ub[4], length=6)
v_grid = range(lb[5], ub[5], length=8)

for T1_5 in T1_5_grid, T6 in T6_grid, T7 in T7_grid, T8_9 in T8_9_grid, v_cm_min in v_grid
    area = objective_area_with_sym_constraint([T1_5, T6, T7, T8_9, v_cm_min], 1.0)
    if area < best_area_base
        best_area_base = area
        best_x_base = [T1_5, T6, T7, T8_9, v_cm_min]
    end
end

# 分析基准的对称性
v_base = best_x_base[5] / 60.0
F_base = [25.0, best_x_base[1], best_x_base[2], best_x_base[3], best_x_base[4], 25.0]
total_time_base = total_length / v_base
t_out_base, _ = solve_heat_equation(a2_vals, h_vals, F_base, v_base, total_time_base)
sigma_base, total_area_base = compute_symmetry(t_out_base)

println("基准解 (问题3风格):")
println("  T1_5=$(round(best_x_base[1], digits=1)), T6=$(round(best_x_base[2], digits=1)), " *
        "T7=$(round(best_x_base[3], digits=1)), T8_9=$(round(best_x_base[4], digits=1)), " *
        "v=$(round(best_x_base[5], digits=2))")
println("  面积: $(round(best_area_base, digits=2)), 对称性σ: $(round(sigma_base, digits=4))")

# 阶段2: 增加对称性约束，逐步收紧
println("\n阶段2: 对称性约束优化")

sigma_targets = [0.3, 0.2, 0.15, 0.1, 0.08]
best_results = []

for sigma_target in sigma_targets
    best_area = Inf
    best_x_sym = zeros(5)

    # 细网格搜索
    T1_5_grid = range(lb[1], ub[1], length=7)
    T6_grid = range(lb[2], ub[2], length=7)
    T7_grid = range(lb[3], ub[3], length=7)
    T8_9_grid = range(lb[4], ub[4], length=7)
    v_grid = range(lb[5], ub[5], length=9)

    for T1_5 in T1_5_grid, T6 in T6_grid, T7 in T7_grid, T8_9 in T8_9_grid, v_cm_min in v_grid
        area = objective_area_with_sym_constraint([T1_5, T6, T7, T8_9, v_cm_min], sigma_target)
        if area < best_area
            best_area = area
            best_x_sym = [T1_5, T6, T7, T8_9, v_cm_min]
        end
    end

    if best_area < Inf
        v_sym = best_x_sym[5] / 60.0
        F_sym = [25.0, best_x_sym[1], best_x_sym[2], best_x_sym[3], best_x_sym[4], 25.0]
        total_time_sym = total_length / v_sym
        t_out_sym, _ = solve_heat_equation(a2_vals, h_vals, F_sym, v_sym, total_time_sym)
        sigma_sym, _ = compute_symmetry(t_out_sym)
        metrics_sym = analyze_curve(t_out_sym)

        push!(best_results, (sigma_target, best_x_sym, best_area, sigma_sym, metrics_sym))
        println("  σ≤$sigma_target: 面积=$(round(best_area, digits=2)), " *
                "实际σ=$(round(sigma_sym, digits=4)), " *
                "T1_5=$(round(best_x_sym[1], digits=1)), T6=$(round(best_x_sym[2], digits=1)), " *
                "T7=$(round(best_x_sym[3], digits=1)), T8_9=$(round(best_x_sym[4], digits=1)), " *
                "v=$(round(best_x_sym[5], digits=2))")
    else
        println("  σ≤$sigma_target: 无可行解")
    end
end

# 最终最优解 (使用最小的σ有解的结果)
if !isempty(best_results)
    final = best_results[end]
    sigma_target, best_x_final, area_final, sigma_final, metrics_final = final

    v_opt = best_x_final[5] / 60.0
    F_opt = [25.0, best_x_final[1], best_x_final[2], best_x_final[3], best_x_final[4], 25.0]
    total_time = total_length / v_opt
    t_out, T_oven = solve_heat_equation(a2_vals, h_vals, F_opt, v_opt, total_time)

    println("\n" * "="^60)
    println("问题4 最终结果")
    println("="^60)
    println("最优温区设定:")
    println("  小温区1-5:  $(round(best_x_final[1], digits=1)) °C")
    println("  小温区6:    $(round(best_x_final[2], digits=1)) °C")
    println("  小温区7:    $(round(best_x_final[3], digits=1)) °C")
    println("  小温区8-9:  $(round(best_x_final[4], digits=1)) °C")
    println("  小温区10-11: 25 °C")
    println("传送带速度: $(round(best_x_final[5], digits=2)) cm/min")
    println("\n炉温曲线指标:")
    println("  超过217°C到峰值的面积: $(round(area_final, digits=2)) °C·s")
    println("  对称性指标 σ: $(round(sigma_final, digits=4))")
    println("  峰值温度: $(round(metrics_final.Tmax, digits=2)) °C")
    println("  最大斜率: $(round(metrics_final.max_slope, digits=3)) °C/s")
    println("  150-190°C (升温)时间: $(round(metrics_final.t_150_190, digits=1)) s")
    println("  >217°C 时间: $(round(metrics_final.t_above_217, digits=1)) s")

    # 计算左右两侧详细数据
    T_sensor = t_out[t_out .>= T_sensor_start]
    above = T_sensor[T_sensor .>= 217] .- 217
    peak_idx = argmax(above)
    t_raw = collect(0:dt:total_time)
    start_idx = findfirst(t -> t >= T_sensor_start, t_out)

    A_L = if peak_idx > 1
        (sum(above[1:peak_idx]) - (above[1] + above[peak_idx])/2) * dt
    else
        above[1] * dt / 2
    end

    A_R = if length(above) - peak_idx > 1
        (sum(above[peak_idx:end]) - (above[peak_idx] + above[end])/2) * dt
    else
        above[end] * dt / 2
    end

    println("  左侧面积 A_L: $(round(A_L, digits=2)) °C·s")
    println("  右侧面积 A_R: $(round(A_R, digits=2)) °C·s")
    println("  峰值位置时间: $(round(t_raw[start_idx+peak_idx-1], digits=1)) s")

    # 绘制最优炉温曲线
    figure(figsize=(14, 6))

    subplot(1, 2, 1)
    plot(t_raw, T_oven, "b-", linewidth=1, label="炉内环境温度")
    plot(t_raw, t_out, "r-", linewidth=1.5, label="焊接区域中心温度")
    axhline(y=217, color="g", linestyle="--", linewidth=0.8, label="T=217°C")
    axhline(y=30, color="gray", linestyle=":", linewidth=0.5, label="T=30°C")

    # 标注峰值
    peak_temp_idx = argmax(t_out)
    scatter([t_raw[peak_temp_idx]], [t_out[peak_temp_idx]], color="red", s=80, zorder=5)
    text(t_raw[peak_temp_idx], t_out[peak_temp_idx] + 10, "峰值 $(round(t_out[peak_temp_idx], digits=1))°C",
         fontsize=8, ha="center")

    # 填充面积
    above_all = copy(t_out)
    above_all[t_out .< 217] .= 217
    fill_between(t_raw[1:peak_temp_idx], 217*ones(peak_temp_idx), above_all[1:peak_temp_idx],
                 alpha=0.3, color="red", label="左侧面积")
    fill_between(t_raw[peak_temp_idx:end], 217*ones(length(t_raw)-peak_temp_idx+1),
                 above_all[peak_temp_idx:end], alpha=0.3, color="blue", label="右侧面积")

    xlabel("时间 (s)")
    ylabel("温度 (°C)")
    title("问题4: 带对称性约束的最优炉温曲线")
    legend(fontsize=7)
    grid(true)

    subplot(1, 2, 2)
    # 对称性分析 - 以峰值为中心的镜像对比
    T_sensor_plot = t_out[t_out .>= T_sensor_start]
    t_sensor_plot = t_raw[findfirst(t -> t >= T_sensor_start, t_out):end]
    peak_idx_sensor = argmax(T_sensor_plot)

    # 左侧曲线
    t_left = t_sensor_plot[1:peak_idx_sensor] .- t_sensor_plot[peak_idx_sensor]
    T_left = T_sensor_plot[1:peak_idx_sensor]

    # 右侧曲线
    t_right = t_sensor_plot[peak_idx_sensor:end] .- t_sensor_plot[peak_idx_sensor]
    T_right = T_sensor_plot[peak_idx_sensor:end]

    # 右侧镜像
    t_right_mirror = -t_right

    plot(t_left, T_left, "r-", linewidth=1.5, label="峰值左侧")
    plot(t_right_mirror, T_right, "b--", linewidth=1.5, label="峰值右侧(镜像)")
    axhline(y=217, color="g", linestyle="--", linewidth=0.8)
    xlabel("相对峰值时间 (s)")
    ylabel("温度 (°C)")
    title("对称性分析 (σ=$(round(sigma_final, digits=4)))")
    legend(fontsize=7)
    grid(true)

    tight_layout()
    savefig("question4_optimal_curve.png", dpi=150)
    println("\n最优炉温曲线图已保存为 question4_optimal_curve.png")

    # 对比问题3和问题4
    println("\n" * "-"^60)
    println("问题3 vs 问题4 对比:")
    println("  问题3 面积: $(round(best_area_base, digits=2)) °C·s, 对称性σ: $(round(sigma_base, digits=4))")
    println("  问题4 面积: $(round(area_final, digits=2)) °C·s, 对称性σ: $(round(sigma_final, digits=4))")
end

println("\n问题4求解完成！")
