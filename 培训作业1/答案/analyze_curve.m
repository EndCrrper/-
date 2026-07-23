function m = analyze_curve(T)
% 提取炉温曲线指标
% T: 温度序列 (已从 T>=30 截取)

    dt = 0.5;
    T = T(:);

    [m.Tmax, m.peak_idx] = max(T);
    slopes = abs(T(2:end) - T(1:end-1)) / dt;
    m.max_slope = max(slopes);

    % 150~190°C 升温段时间 (计数间隔)
    rising = [false; T(2:end) >= T(1:end-1)];
    in_range = (T >= 150) & (T <= 190);
    n = sum(in_range & rising);
    m.t_150_190 = max(0, n - 1) * dt;

    % >217°C 时间 (计数间隔)
    n = sum(T >= 217);
    m.t_above = max(0, n - 1) * dt;

    % 超过217°C到峰值的面积 (梯形积分)
    above = T - 217;
    above = above(above >= 0);    % 仅保留 T >= 217 的部分
    if isempty(above)
        m.area = Inf;
    else
        [~, k] = max(above);      % 峰值在 above 序列中的位置
        if k > 1
            m.area = (sum(above(1:k)) - (above(1)+above(k))/2) * dt;
        else
            m.area = above(1) * dt / 2;
        end
    end
end
