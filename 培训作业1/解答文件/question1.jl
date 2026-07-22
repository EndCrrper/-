#=
问题1: 给定传送带速度和各温区温度设定值，计算焊接区域中心温度变化
v = 78 cm/min, T = [173, 173, 173, 173, 173, 198, 230, 257, 257, 25, 25]
=#

include("solution_core.jl")

# ==================== 热力学参数 (参考论文优化结果) ====================
# 10个参数: [a1², h1, a2², h2, a3², h3, a4², h4, a5², h5]
const xm = [6.677003439269690e-04; 2.999997701878186e+04;
            8.120630727730775e-04; 1.499997442761376e+03;
            9.300295943899003e-04; 1.389867556248244e+03;
            8.343291166559698e-04; 6.898624109710258e+02;
            5.307244068332524e-04; 1.265962465478697e+03]

# ==================== 问题1参数 ====================
v1 = 78.0 / 60.0  # cm/s
F1 = [25.0, 173.0, 198.0, 230.0, 257.0, 25.0]
total_time1 = total_length / v1

println("\n" * "="^60)
println("问题1: 给定参数下的炉温曲线计算")
println("="^60)
println("传送带速度: $(78.0) cm/min = $(round(v1, digits=4)) cm/s")
println("各温区温度: 1-5: $(F1[2])°C, 6: $(F1[3])°C, 7: $(F1[4])°C, 8-9: $(F1[5])°C, 10-11: 25°C")
println("总仿真时间: $(round(total_time1, digits=1)) s")

# 求解
a2_vals = [xm[1], xm[3], xm[5], xm[7], xm[9]]
h_vals = [xm[2], xm[4], xm[6], xm[8], xm[10]]

t_all, T_oven = solve_heat_equation(a2_vals, h_vals, F1, v1, total_time1)
t_raw = collect(0:dt:total_time1)
T_center = t_all  # 焊接区域中心温度

# 截取从传感器启动的温度数据
start_idx = findfirst(t -> t >= T_sensor_start, T_center)
t_sensor = t_raw[start_idx:end]
T_sensor = T_center[start_idx:end]

println("\n温度传感器数据范围: t = $(t_sensor[1]) ~ $(t_sensor[end]) s")
println("温度数据点数: $(length(t_sensor))")

# 计算指定位置的温度
# 各关键位置 (cm):
# 温区3中点: 前区(25) + 2*(温区+间隙) + 温区/2 = 25 + 2*35.5 + 15.25 = 111.25
# 温区6中点: 前区(25) + 5*温区 + 4*间隙 + 间隙 + 温区/2 = 25+152.5+20+5+15.25 = 217.75
# 温区7中点: 25+5*30.5+4*5+5+30.5+5+15.25 = 25+152.5+20+5+30.5+5+15.25 = 253.25
# 温区8结束处: 25+5*30.5+4*5+5+30.5+5+30.5+5+30.5 = 25+152.5+20+5+30.5+5+30.5+5+30.5 = 304

pos_z3_mid = front_length + 2*(zone_length + gap_length) + zone_length/2
pos_z6_mid = front_length + 5*zone_length + 4*gap_length + gap_length + zone_length/2
pos_z7_mid = front_length + 5*zone_length + 4*gap_length + gap_length + zone_length + gap_length + zone_length/2
pos_z8_end = front_length + 5*zone_length + 4*gap_length + gap_length + zone_length + gap_length + zone_length + gap_length + zone_length

# 对应时间
t_z3 = pos_z3_mid / v1
t_z6 = pos_z6_mid / v1
t_z7 = pos_z7_mid / v1
t_z8 = pos_z8_end / v1

# 插值求温度
function interp_temp(t_target, t_vec, T_vec)
    if t_target <= t_vec[1]
        return T_vec[1]
    elseif t_target >= t_vec[end]
        return T_vec[end]
    end
    idx = findlast(t -> t <= t_target, t_vec)
    if idx === nothing || idx == length(t_vec)
        return T_vec[end]
    end
    # 线性插值
    alpha = (t_target - t_vec[idx]) / (t_vec[idx+1] - t_vec[idx])
    return T_vec[idx] + alpha * (T_vec[idx+1] - T_vec[idx])
end

T_z3 = interp_temp(t_z3, t_raw, T_center)
T_z6 = interp_temp(t_z6, t_raw, T_center)
T_z7 = interp_temp(t_z7, t_raw, T_center)
T_z8 = interp_temp(t_z8, t_raw, T_center)

println("\n指定位置温度:")
println("  温区3中点 (x=$(round(pos_z3_mid, digits=1)) cm, t=$(round(t_z3, digits=1)) s): $(round(T_z3, digits=2)) °C")
println("  温区6中点 (x=$(round(pos_z6_mid, digits=1)) cm, t=$(round(t_z6, digits=1)) s): $(round(T_z6, digits=2)) °C")
println("  温区7中点 (x=$(round(pos_z7_mid, digits=1)) cm, t=$(round(t_z7, digits=1)) s): $(round(T_z7, digits=2)) °C")
println("  温区8结束 (x=$(round(pos_z8_end, digits=1)) cm, t=$(round(t_z8, digits=1)) s): $(round(T_z8, digits=2)) °C")

# 曲线分析
metrics = analyze_curve(T_center)
println("\n炉温曲线指标:")
println("  峰值温度: $(round(metrics.Tmax, digits=2)) °C")
println("  最大斜率: $(round(metrics.max_slope, digits=2)) °C/s")
println("  150-190°C (升温)时间: $(round(metrics.t_150_190, digits=1)) s")
println("  >217°C 时间: $(round(metrics.t_above_217, digits=1)) s")
println("  超过217°C到峰值的面积: $(round(metrics.area, digits=2)) °C·s")

# 生成 result.csv - 每0.5s的温度数据
# 传感器从温度达到30°C时开始工作
start_idx_csv = findfirst(t -> t >= 30.0, T_center)
if start_idx_csv !== nothing
    result_data = hcat(t_raw[start_idx_csv:end], T_center[start_idx_csv:end])
else
    result_data = hcat(t_raw, T_center)
end

# 写入CSV文件
open("result.csv", "w") do io
    write(io, "时间(s),温度(°C)\n")
    for i in 1:size(result_data, 1)
        write(io, "$(round(result_data[i,1], digits=1)),$(round(result_data[i,2], digits=4))\n")
    end
end
println("\nresult.csv 已生成，共 $(size(result_data,1)) 行数据")

# 绘制炉温曲线
figure(figsize=(12, 5))

subplot(1, 2, 1)
plot(t_raw, T_oven, "b-", linewidth=1, label="炉内环境温度")
plot(t_raw, T_center, "r-", linewidth=1.5, label="焊接区域中心温度")
# 标注217°C线
axhline(y=217, color="g", linestyle="--", linewidth=0.8, label="T=217°C")
# 标注30°C线
axhline(y=30, color="gray", linestyle=":", linewidth=0.5, label="T=30°C")

# 标注关键点
scatter([t_z3], [T_z3], color="orange", s=50, zorder=5)
text(t_z3, T_z3-15, "温区3中点", fontsize=8, ha="center")
scatter([t_z6], [T_z6], color="orange", s=50, zorder=5)
text(t_z6, T_z6-15, "温区6中点", fontsize=8, ha="center")
scatter([t_z7], [T_z7], color="orange", s=50, zorder=5)
text(t_z7, T_z7-15, "温区7中点", fontsize=8, ha="center")
scatter([t_z8], [T_z8], color="orange", s=50, zorder=5)
text(t_z8, T_z8-15, "温区8结束", fontsize=8, ha="center")

xlabel("时间 (s)")
ylabel("温度 (°C)")
title("问题1: 炉温曲线")
legend(fontsize=7)
grid(true)

subplot(1, 2, 2)
# 距离-温度图
x_pos = v1 .* t_raw
plot(x_pos, T_center, "r-", linewidth=1.5, label="焊接区域中心温度")
plot(x_pos, T_oven, "b--", linewidth=1, label="炉内环境温度")
# 标注温区边界
for (i, temp) in enumerate([173, 173, 173, 173, 173, 198, 230, 257, 257, 25, 25])
    x_start = front_length + (i-1)*(zone_length + gap_length)
    if i > 5 && i <= 9
        x_start = front_length + 5*zone_length + 4*gap_length
        if i == 6
            x_start += gap_length
        elseif i == 7
            x_start += gap_length + zone_length + gap_length
        elseif i == 8
            x_start += gap_length + zone_length + gap_length + zone_length + gap_length
        elseif i == 9
            x_start = front_length + 5*zone_length + 4*gap_length + gap_length + zone_length + gap_length + zone_length + gap_length + zone_length
        end
    end
    # simplified - draw zone regions
end
xlabel("位置 (cm)")
ylabel("温度 (°C)")
title("位置-温度曲线")
legend(fontsize=7)
grid(true)

tight_layout()
savefig("question1_curve.png", dpi=150)
println("炉温曲线图已保存为 question1_curve.png")

println("\n问题1求解完成！")
