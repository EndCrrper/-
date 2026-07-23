function m = analyze_curve(T)
% 提取炉温曲线指标
% T: 温度序列 (已从 T>=30 截取)

    dt = 0.5;
    T = T(:);

    [m.Tmax, m.peak_idx] = max(T);
    slopes = abs(T(2:end) - T(1:end-1)) / dt;
    m.max_slope = max(slopes);

    % 150~190°C 升温段时间
    rising = [false; T(2:end) >= T(1:end-1)];
    in_range = (T >= 150) & (T <= 190);
    m.t_150_190 = sum(in_range & rising) * dt;

    % >217°C 时间
    m.t_above = sum(T >= 217) * dt;

    % 超过217°C到峰值的面积 (梯形积分)
    above = T - 217;
    above = above(above > 0);
    if isempty(above)
        m.area = Inf;
    else
        k = find(above == max(above), 1);
        if k > 1
            m.area = (sum(above(1:k)) - (above(1)+above(k))/2) * dt;
        else
            m.area = above(1) * dt / 2;
        end
    end
end
