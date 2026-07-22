#=
问题3: 优化炉温曲线，使超过217°C到峰值温度的面积最小
决策变量: T1_5 ∈ [165, 185], T6 ∈ [185, 205], T7 ∈ [225, 245], T8_9 ∈ [245, 265], v ∈ [65, 100]
约束: 满足制程界限
目标: 最小化面积 S = ∫(T(t)-217)dt 从217°C交点到峰值
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
println("问题3: 最小化超过217°C到峰值的面积")
println("="^60)

# 目标函数: 计算面积(不满足约束返回Inf)
function objective_area(x)
    T1_5, T6, T7, T8_9, v_cm_min = x
    v = v_cm_min / 60.0
    F = [25.0, T1_5, T6, T7, T8_9, 25.0]
    total_time = total_length / v

    t_out, _ = solve_heat_equation(a2_vals, h_vals, F, v, total_time)
    T = t_out[t_out .>= T_sensor_start]

    # 检查制程界限
    metrics = analyze_curve(t_out)
    if !check_constraints(metrics)
        return Inf
    end

    return metrics.area
end

# 精确计算面积 (用于最终结果)
function compute_area_detailed(T)
    above = T .- 217
    above = above[above .> 0]
    if isempty(above)
        return 0.0
    end
    peak_idx = argmax(above)
    if peak_idx > 1
        area = (sum(above[1:peak_idx]) - (above[1] + above[peak_idx])/2) * dt
    else
        area = (above[1] + above[peak_idx]) * dt / 2
    end
    return area
end

# 决策变量边界: [T1_5, T6, T7, T8_9, v]
# 各温区温度可调整±10°C (问题中未明确给出, 根据行业标准为±10°C)
# T1_5: 175±10 → [165, 185]
# T6: 195±10 → [185, 205]
# T7: 235±10 → [225, 245]
# T8_9: 255±10 → [245, 265]
lb = [165.0, 185.0, 225.0, 245.0, 65.0]
ub = [185.0, 205.0, 245.0, 265.0, 100.0]

println("决策变量:")
println("  T1_5 ∈ [$(lb[1]), $(ub[1])] °C")
println("  T6   ∈ [$(lb[2]), $(ub[2])] °C")
println("  T7   ∈ [$(lb[3]), $(ub[3])] °C")
println("  T8_9 ∈ [$(lb[4]), $(ub[4])] °C")
println("  v    ∈ [$(lb[5]), $(ub[5])] cm/min")

# 网格搜索 - 粗搜索
println("\n阶段1: 粗网格搜索...")

best_area = Inf
best_x = zeros(5)
grid_size = [5, 5, 5, 5, 8]  # 各维度网格点数

T1_5_grid = range(lb[1], ub[1], length=grid_size[1])
T6_grid = range(lb[2], ub[2], length=grid_size[2])
T7_grid = range(lb[3], ub[3], length=grid_size[3])
T8_9_grid = range(lb[4], ub[4], length=grid_size[4])
v_grid = range(lb[5], ub[5], length=grid_size[5])

total_combinations = prod(grid_size)
println("总搜索组合数: $total_combinations")

count = 0
for T1_5 in T1_5_grid, T6 in T6_grid, T7 in T7_grid, T8_9 in T8_9_grid, v_cm_min in v_grid
    count += 1
    area = objective_area([T1_5, T6, T7, T8_9, v_cm_min])
    if area < best_area
        best_area = area
        best_x = [T1_5, T6, T7, T8_9, v_cm_min]
    end
end

println("粗搜索完成!")
println("最优解: T1_5=$(round(best_x[1], digits=1))°C, T6=$(round(best_x[2], digits=1))°C, " *
        "T7=$(round(best_x[3], digits=1))°C, T8_9=$(round(best_x[4], digits=1))°C, " *
        "v=$(round(best_x[5], digits=1)) cm/min")
println("最小面积: $(round(best_area, digits=2)) °C·s")

# 阶段2: 局部精细搜索 (在最优解附近)
println("\n阶段2: 局部精细搜索...")

fine_range_T = 2.0  # 温度搜索范围 ±2°C
fine_range_v = 2.0   # 速度搜索范围 ±2 cm/min

T1_5_fine = range(max(lb[1], best_x[1]-fine_range_T), min(ub[1], best_x[1]+fine_range_T), length=7)
T6_fine = range(max(lb[2], best_x[2]-fine_range_T), min(ub[2], best_x[2]+fine_range_T), length=7)
T7_fine = range(max(lb[3], best_x[3]-fine_range_T), min(ub[3], best_x[3]+fine_range_T), length=7)
T8_9_fine = range(max(lb[4], best_x[4]-fine_range_T), min(ub[4], best_x[4]+fine_range_T), length=7)
v_fine = range(max(lb[5], best_x[5]-fine_range_v), min(ub[5], best_x[5]+fine_range_v), length=11)

count2 = 0
for T1_5 in T1_5_fine, T6 in T6_fine, T7 in T7_fine, T8_9 in T8_9_fine, v_cm_min in v_fine
    count2 += 1
    area = objective_area([T1_5, T6, T7, T8_9, v_cm_min])
    if area < best_area
        best_area = area
        best_x = [T1_5, T6, T7, T8_9, v_cm_min]
    end
end

println("精细搜索完成 (搜索 $(count2) 个点)")

# 最优解分析
v_opt = best_x[5] / 60.0
F_opt = [25.0, best_x[1], best_x[2], best_x[3], best_x[4], 25.0]
total_time = total_length / v_opt
t_out, T_oven = solve_heat_equation(a2_vals, h_vals, F_opt, v_opt, total_time)
metrics = analyze_curve(t_out)

# 从传感器启动截取
T_sensor = t_out[t_out .>= T_sensor_start]
t_raw = collect(0:dt:total_time)
start_idx = findfirst(t -> t >= T_sensor_start, t_out)
t_sensor = t_raw[start_idx:start_idx+length(T_sensor)-1]

println("\n" * "="^60)
println("问题3 最终结果")
println("="^60)
println("最优温区设定:")
println("  小温区1-5:  $(round(best_x[1], digits=1)) °C")
println("  小温区6:    $(round(best_x[2], digits=1)) °C")
println("  小温区7:    $(round(best_x[3], digits=1)) °C")
println("  小温区8-9:  $(round(best_x[4], digits=1)) °C")
println("  小温区10-11: 25 °C")
println("传送带速度: $(round(best_x[5], digits=2)) cm/min")
println("\n炉温曲线指标:")
println("  超过217°C到峰值的面积: $(round(best_area, digits=2)) °C·s")
println("  峰值温度: $(round(metrics.Tmax, digits=2)) °C")
println("  最大斜率: $(round(metrics.max_slope, digits=3)) °C/s")
println("  150-190°C (升温)时间: $(round(metrics.t_150_190, digits=1)) s")
println("  >217°C 时间: $(round(metrics.t_above_217, digits=1)) s")

# 绘制最优炉温曲线
figure(figsize=(12, 5))

subplot(1, 2, 1)
plot(t_raw, T_oven, "b-", linewidth=1, label="炉内环境温度")
plot(t_raw, t_out, "r-", linewidth=1.5, label="焊接区域中心温度")
axhline(y=217, color="g", linestyle="--", linewidth=0.8, label="T=217°C")
axhline(y=30, color="gray", linestyle=":", linewidth=0.5, label="T=30°C")

# 填充面积
above = copy(t_out)
above[t_out .< 217] .= 217
peak_idx = argmax(t_out[t_out .>= T_sensor_start]) + (start_idx - 1)
fill_between(t_raw[1:peak_idx], 217*ones(peak_idx), above[1:peak_idx], alpha=0.3, color="red")

xlabel("时间 (s)")
ylabel("温度 (°C)")
title("问题3: 最优炉温曲线")
legend(fontsize=7)
grid(true)

subplot(1, 2, 2)
x_pos = v_opt .* t_raw
plot(x_pos, t_out, "r-", linewidth=1.5, label="焊接区域中心温度")
plot(x_pos, T_oven, "b--", linewidth=1, label="炉内环境温度")
axhline(y=217, color="g", linestyle="--", linewidth=0.8, label="T=217°C")
xlabel("位置 (cm)")
ylabel("温度 (°C)")
title("位置-温度曲线")
legend(fontsize=7)
grid(true)

tight_layout()
savefig("question3_optimal_curve.png", dpi=150)
println("\n最优炉温曲线图已保存为 question3_optimal_curve.png")

println("\n问题3求解完成！")
