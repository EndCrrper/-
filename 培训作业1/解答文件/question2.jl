#=
问题2: 确定允许的最大传送带过炉速度
各温区温度: T = [182, 182, 182, 182, 182, 203, 237, 254, 254, 25, 25]
速度范围: 65-100 cm/min
约束: 满足制程界限
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

# 问题2的温区设定
F2 = [25.0, 182.0, 203.0, 237.0, 254.0, 25.0]

println("\n" * "="^60)
println("问题2: 确定允许的最大传送带过炉速度")
println("="^60)
println("各温区温度: 1-5: $(F2[2])°C, 6: $(F2[3])°C, 7: $(F2[4])°C, 8-9: $(F2[5])°C")

# 先扫描不同速度下的指标
println("\n扫描不同速度下的炉温曲线指标:")
println("-"^80)
println("速度(cm/min) | 峰值(°C) | 最大斜率(°C/s) | 150-190°C(s) | >217°C(s) | 满足约束")
println("-"^80)

v_range = 65:5:100
scan_results = []
for v_cm_min in v_range
    v = v_cm_min / 60.0
    total_time = total_length / v
    t_out, _ = solve_heat_equation(a2_vals, h_vals, F2, v, total_time)
    metrics = analyze_curve(t_out)
    ok = check_constraints(metrics)
    push!(scan_results, (v_cm_min, metrics, ok))
    println("$(lpad(v_cm_min, 4))          | $(round(metrics.Tmax, digits=1))     | $(round(metrics.max_slope, digits=2))          | $(round(metrics.t_150_190, digits=1))        | $(round(metrics.t_above_217, digits=1))       | $ok")
end

# 二分法精确搜索最大速度
function is_feasible(v_cm_min)
    v = v_cm_min / 60.0
    total_time = total_length / v
    t_out, _ = solve_heat_equation(a2_vals, h_vals, F2, v, total_time)
    metrics = analyze_curve(t_out)
    return check_constraints(metrics)
end

# 找到可行的速度区间
feasible_mask = [r[3] for r in scan_results]
feasible_v = [r[1] for r in scan_results[feasible_mask]]

if isempty(feasible_v)
    println("\n警告: 在65-100 cm/min范围内没有找到可行速度!")
    # 放宽检查: 找出最接近满足约束的速度
else
    v_low = minimum(feasible_v)
    v_high = maximum(feasible_v)

    # 在 [v_high, 100] 区间二分搜索
    v_left = v_high
    v_right = min(v_high + 5.0, 100.0)

    # 先检查是否v_right处也可行
    if is_feasible(v_right)
        v_left = v_right
        v_right = 100.0
    end

    while v_right - v_left > 1e-4
        v_mid = (v_left + v_right) / 2
        if is_feasible(v_mid)
            v_left = v_mid
        else
            v_right = v_mid
        end
    end

    v_max = v_left
    println("\n" * "-"^60)
    println("二分法精确搜索结果:")
    println("  最大允许传送带速度: $(round(v_max, digits=3)) cm/min")

    # 验证最优速度下的结果
    v_opt = v_max / 60.0
    total_time = total_length / v_opt
    t_out, T_oven = solve_heat_equation(a2_vals, h_vals, F2, v_opt, total_time)
    metrics = analyze_curve(t_out)

    println("\n最优速度下的炉温曲线指标:")
    println("  峰值温度: $(round(metrics.Tmax, digits=2)) °C")
    println("  最大斜率: $(round(metrics.max_slope, digits=3)) °C/s")
    println("  150-190°C (升温)时间: $(round(metrics.t_150_190, digits=1)) s")
    println("  >217°C 时间: $(round(metrics.t_above_217, digits=1)) s")
    println("  满足制程界限: $(check_constraints(metrics))")

    # 绘制速度-指标关系图
    figure(figsize=(12, 8))

    subplot(2, 2, 1)
    v_list = [r[1] for r in scan_results]
    plot(v_list, [r[2].max_slope for r in scan_results], "b-o", markersize=6)
    axhline(y=3, color="r", linestyle="--", label="上限 3°C/s")
    xlabel("传送带速度 (cm/min)")
    ylabel("最大斜率 (°C/s)")
    title("最大温度斜率 vs 传送速度")
    legend()
    grid(true)

    subplot(2, 2, 2)
    plot(v_list, [r[2].Tmax for r in scan_results], "r-o", markersize=6)
    axhline(y=250, color="r", linestyle="--", label="上限 250°C")
    axhline(y=240, color="orange", linestyle="--", label="下限 240°C")
    xlabel("传送带速度 (cm/min)")
    ylabel("峰值温度 (°C)")
    title("峰值温度 vs 传送速度")
    legend()
    grid(true)

    subplot(2, 2, 3)
    plot(v_list, [r[2].t_above_217 for r in scan_results], "g-o", markersize=6)
    axhline(y=90, color="r", linestyle="--", label="上限 90s")
    axhline(y=40, color="orange", linestyle="--", label="下限 40s")
    xlabel("传送带速度 (cm/min)")
    ylabel("时间 (s)")
    title(">217°C时间 vs 传送速度")
    legend()
    grid(true)

    subplot(2, 2, 4)
    plot(v_list, [r[2].t_150_190 for r in scan_results], "m-o", markersize=6)
    axhline(y=120, color="r", linestyle="--", label="上限 120s")
    axhline(y=60, color="orange", linestyle="--", label="下限 60s")
    xlabel("传送带速度 (cm/min)")
    ylabel("时间 (s)")
    title("150-190°C时间 vs 传送速度")
    legend()
    grid(true)

    tight_layout()
    savefig("question2_analysis.png", dpi=150)
    println("\n分析图已保存为 question2_analysis.png")
end

println("\n问题2求解完成！")
