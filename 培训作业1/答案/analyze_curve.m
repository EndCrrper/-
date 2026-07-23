function metrics = analyze_curve(T)
% analyze_curve - 分析炉温曲线的各项指标
%
% 输入:
%   T : 焊接区域中心温度序列 (°C)，已从 T>=30°C 截取
%
% 输出:
%   metrics : 结构体，包含以下字段:
%     .Tmax      - 峰值温度 (°C)
%     .max_slope - 最大升/降温速率 (°C/s)
%     .t_150_190 - 升温过程中 150°C~190°C 的时间 (s)
%     .t_above   - 温度大于 217°C 的时间 (s)
%     .area      - 超过 217°C 到峰值温度的面积 (°C·s)
%     .peak_idx  - 峰值温度的索引

    dt = 0.5;
    T = T(:);  % 确保为列向量

    % ---- 峰值温度 ----
    [Tmax, peak_idx] = max(T);

    % ---- 最大升/降温速率 ----
    slopes = abs(T(2:end) - T(1:end-1)) / dt;
    max_slope = max(slopes);

    % ---- 升温过程中 150°C~190°C 的时间 ----
    rising = [false; T(2:end) >= T(1:end-1)];  % 升温标记
    in_range = (T >= 150) & (T <= 190);
    t_150_190 = sum(in_range & rising) * dt;

    % ---- 温度大于 217°C 的时间 ----
    t_above = sum(T >= 217) * dt;

    % ---- 超过 217°C 到峰值温度的面积 (梯形积分) ----
    above_217 = T - 217;
    above_vals = above_217(above_217 > 0);
    peak_in_above = find(above_vals == max(above_vals), 1);

    if isempty(above_vals)
        area = Inf;
    elseif length(above_vals) == 1
        area = above_vals(1) * dt / 2;
    else
        if peak_in_above > 1
            area = (sum(above_vals(1:peak_in_above)) - ...
                    (above_vals(1) + above_vals(peak_in_above)) / 2) * dt;
        else
            area = (above_vals(1) + above_vals(peak_in_above)) * dt / 2;
        end
    end

    % ---- 输出结构体 ----
    metrics = struct(...
        'Tmax',      Tmax, ...
        'max_slope', max_slope, ...
        't_150_190', t_150_190, ...
        't_above',   t_above, ...
        'area',      area, ...
        'peak_idx',  peak_idx);
end
